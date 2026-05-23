//! Runtime backend for `#[napi] async fn` Promise scheduling.
//!
//! Two backends, selected at compile time via Cargo features:
//!
//! * `tokio_rt` (default via the `async` feature) — drives futures on a
//!   `tokio::runtime::Runtime`. Lazily created the first time it's needed and
//!   overridable via `create_custom_tokio_runtime`. Public `spawn`,
//!   `spawn_blocking`, `block_on`, `within_runtime_if_available` map to
//!   `tokio::runtime::Runtime` methods and return `tokio::task::JoinHandle`s.
//!
//! * `runtime_executor` — drives futures on a `futures::executor::ThreadPool`
//!   created lazily on first use. Sync `block_on` is `pollster::block_on`,
//!   `spawn_blocking` runs the closure on its own `std::thread::spawn`. No
//!   tokio dependency. JoinHandle types resolve to a custom wrapper around a
//!   `futures::channel::oneshot::Receiver` so the public surface
//!   (`.await`-on-handle, output forwarding) remains the same shape.
//!
//! The macro-generated calls to `execute_tokio_future_with_finalize_callback`
//! are runtime-agnostic at the call site; only the body of this module
//! changes between backends.

#[cfg(not(feature = "noop"))]
use std::sync::{LazyLock, OnceLock, RwLock};
use std::{future::Future, marker::PhantomData};

#[cfg(feature = "tokio_rt")]
use tokio::runtime::Runtime;

use crate::{bindgen_runtime::ToNapiValue, sys, Env, Error, Result};
#[cfg(not(feature = "noop"))]
use crate::{JsDeferred, SendableResolver, Unknown};

// ---------------------------------------------------------------------------
// `tokio_rt` backend
// ---------------------------------------------------------------------------

#[cfg(all(feature = "tokio_rt", not(feature = "runtime_executor"), not(feature = "noop")))]
fn create_runtime() -> Runtime {
  // Check if we're supposed to use a user-defined runtime
  if IS_USER_DEFINED_RT.get().copied().unwrap_or(false) {
    // Try to take the user-defined runtime if it's still available
    if let Some(user_defined_rt) = USER_DEFINED_RT
      .get()
      .and_then(|rt| rt.write().ok().and_then(|mut rt| rt.take()))
    {
      return user_defined_rt;
    }
    // If the user-defined runtime was already taken, fall back to creating a default runtime
    // This handles the case where the runtime was shutdown and needs to be restarted
  }

  #[cfg(any(
    all(target_family = "wasm", tokio_unstable),
    not(target_family = "wasm")
  ))]
  {
    tokio::runtime::Builder::new_multi_thread()
      .enable_all()
      .build()
      .expect("Create tokio runtime failed")
  }
  #[cfg(all(target_family = "wasm", not(tokio_unstable)))]
  {
    tokio::runtime::Builder::new_current_thread()
      .enable_all()
      .build()
      .expect("Create tokio runtime failed")
  }
}

#[cfg(all(feature = "tokio_rt", not(feature = "runtime_executor"), not(feature = "noop")))]
static RT: LazyLock<RwLock<Option<Runtime>>> =
  LazyLock::new(|| RwLock::new(Some(create_runtime())));

#[cfg(all(feature = "tokio_rt", not(feature = "runtime_executor"), not(feature = "noop")))]
static USER_DEFINED_RT: OnceLock<RwLock<Option<Runtime>>> = OnceLock::new();

#[cfg(all(feature = "tokio_rt", not(feature = "runtime_executor"), not(feature = "noop")))]
static IS_USER_DEFINED_RT: OnceLock<bool> = OnceLock::new();

#[cfg(all(feature = "tokio_rt", not(feature = "runtime_executor"), not(feature = "noop")))]
/// Create a custom Tokio runtime used by the NAPI-RS.
/// You can control the tokio runtime configuration by yourself.
/// ### Example
/// ```no_run
/// use tokio::runtime::Builder;
/// use napi::create_custom_tokio_runtime;
///
/// #[napi_derive::module_init]
/// fn init() {
///    let rt = Builder::new_multi_thread().enable_all().thread_stack_size(32 * 1024 * 1024).build().unwrap();
///    create_custom_tokio_runtime(rt);
/// }
pub fn create_custom_tokio_runtime(rt: Runtime) {
  USER_DEFINED_RT.get_or_init(move || RwLock::new(Some(rt)));
  IS_USER_DEFINED_RT.get_or_init(|| true);
}

#[cfg(all(feature = "tokio_rt", not(feature = "runtime_executor"), feature = "noop"))]
pub fn create_custom_tokio_runtime(_: Runtime) {}

#[cfg(all(feature = "tokio_rt", not(feature = "runtime_executor"), not(feature = "noop")))]
/// Start the async runtime (Currently is tokio).
///
/// In Node.js native targets the async runtime will be dropped when Node env exits.
/// But in Electron renderer process, the Node env will exits and recreate when the window reloads.
/// So we need to ensure that the async runtime is initialized when the Node env is created.
///
/// In wasm targets, the async runtime will not been shutdown automatically due to the limitation of the wasm runtime.
/// So, you need to call `shutdown_async_runtime` function to manually shutdown the async runtime.
/// In some scenarios, you may want to start the async runtime again like in tests.
pub fn start_async_runtime() {
  if let Ok(mut rt) = RT.write() {
    if rt.is_none() {
      *rt = Some(create_runtime());
    }
  }
}

#[cfg(all(feature = "tokio_rt", not(feature = "runtime_executor"), not(feature = "noop")))]
pub fn shutdown_async_runtime() {
  if let Some(rt) = RT.write().ok().and_then(|mut rt| rt.take()) {
    rt.shutdown_background();
  }
}

#[cfg(all(feature = "tokio_rt", not(feature = "runtime_executor"), not(feature = "noop")))]
/// Spawns a future onto the Tokio runtime.
///
/// Depending on where you use it, you should await or abort the future in your drop function.
/// To avoid undefined behavior and memory corruptions.
pub fn spawn<F>(fut: F) -> tokio::task::JoinHandle<F::Output>
where
  F: 'static + Send + Future<Output = ()>,
{
  RT.read()
    .ok()
    .and_then(|rt| rt.as_ref().map(|rt| rt.spawn(fut)))
    .expect("Access tokio runtime failed in spawn")
}

#[cfg(all(feature = "tokio_rt", not(feature = "runtime_executor"), not(feature = "noop")))]
/// Runs a future to completion
/// This is blocking, meaning that it pauses other execution until the future is complete,
/// only use it when it is absolutely necessary, in other places use async functions instead.
pub fn block_on<F: Future>(fut: F) -> F::Output {
  RT.read()
    .ok()
    .and_then(|rt| rt.as_ref().map(|rt| rt.block_on(fut)))
    .expect("Access tokio runtime failed in block_on")
}

#[cfg(all(feature = "tokio_rt", not(feature = "runtime_executor"), not(feature = "noop")))]
/// spawn_blocking on the current Tokio runtime.
pub fn spawn_blocking<F, R>(func: F) -> tokio::task::JoinHandle<R>
where
  F: FnOnce() -> R + Send + 'static,
  R: Send + 'static,
{
  RT.read()
    .ok()
    .and_then(|rt| rt.as_ref().map(|rt| rt.spawn_blocking(func)))
    .expect("Access tokio runtime failed in spawn_blocking")
}

#[cfg(all(feature = "tokio_rt", not(feature = "runtime_executor"), not(feature = "noop")))]
/// If the feature `tokio_rt` has been enabled this will enter the runtime context and
/// then call the provided closure. Otherwise it will just call the provided closure.
pub fn within_runtime_if_available<F: FnOnce() -> T, T>(f: F) -> T {
  RT.read()
    .ok()
    .and_then(|rt| {
      rt.as_ref().map(|rt| {
        let rt_guard = rt.enter();
        let ret = f();
        drop(rt_guard);
        ret
      })
    })
    .expect("Access tokio runtime failed in within_runtime_if_available")
}

// ---------------------------------------------------------------------------
// `runtime_executor` backend
// ---------------------------------------------------------------------------

#[cfg(all(feature = "runtime_executor", not(feature = "noop")))]
mod executor_backend {
  use futures::executor::ThreadPool;
  use std::future::Future;
  use std::sync::OnceLock;

  /// Process-wide `futures::executor::ThreadPool`, created lazily on first
  /// use. We don't expose configuration knobs (yet) — `ThreadPool::new()`
  /// defaults to `num_cpus::get()` threads, which is the right default for
  /// driving short-lived Promise futures whose bodies hand actual work off
  /// to rayon or other internal pools.
  pub(super) fn pool() -> &'static ThreadPool {
    static POOL: OnceLock<ThreadPool> = OnceLock::new();
    POOL.get_or_init(|| ThreadPool::new().expect("failed to create napi runtime ThreadPool"))
  }

  /// `runtime_executor`'s answer to `tokio::task::JoinHandle`. The handle
  /// resolves to `F::Output` and exposes panic-aware completion via the
  /// inner oneshot — matching tokio's `.await` shape closely enough for
  /// napi's internal use.
  pub struct JoinHandle<T> {
    rx: futures::channel::oneshot::Receiver<Result<T, Box<dyn std::any::Any + Send>>>,
  }

  impl<T: Send + 'static> JoinHandle<T> {
    pub(super) fn spawn<F: Future<Output = T> + Send + 'static>(fut: F) -> Self {
      let (tx, rx) = futures::channel::oneshot::channel();
      pool().spawn_ok(async move {
        // Spawn each task with panic catching so we can mirror tokio's
        // JoinError semantics.
        let result = std::panic::AssertUnwindSafe(fut);
        let outcome = match futures::FutureExt::catch_unwind(result).await {
          Ok(v) => Ok(v),
          Err(panic) => Err(panic),
        };
        let _ = tx.send(outcome);
      });
      Self { rx }
    }
  }

  impl<T> std::future::Future for JoinHandle<T> {
    type Output = std::result::Result<T, JoinError>;

    fn poll(
      mut self: std::pin::Pin<&mut Self>,
      cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
      use std::task::Poll;
      let rx = std::pin::Pin::new(&mut self.rx);
      match futures::Future::poll(rx, cx) {
        Poll::Pending => Poll::Pending,
        Poll::Ready(Err(_canceled)) => {
          // Sender dropped without sending — treat as a generic cancel.
          Poll::Ready(Err(JoinError { panic: None }))
        }
        Poll::Ready(Ok(Ok(v))) => Poll::Ready(Ok(v)),
        Poll::Ready(Ok(Err(panic))) => Poll::Ready(Err(JoinError {
          panic: Some(panic),
        })),
      }
    }
  }

  /// Mirror of `tokio::task::JoinError` shaped enough for napi's internal
  /// "try_into_panic"-style error handling in `execute_tokio_future*`.
  pub struct JoinError {
    panic: Option<Box<dyn std::any::Any + Send>>,
  }

  impl JoinError {
    pub fn try_into_panic(self) -> std::result::Result<Box<dyn std::any::Any + Send>, Self> {
      match self.panic {
        Some(p) => Ok(p),
        None => Err(self),
      }
    }
  }
}

#[cfg(all(feature = "runtime_executor", not(feature = "noop")))]
pub use executor_backend::{JoinError, JoinHandle};

#[cfg(all(feature = "runtime_executor", not(feature = "noop")))]
/// Spawn a future onto the global ThreadPool. Returns a `JoinHandle` whose
/// `.await` resolves to `Result<F::Output, JoinError>` (the panic-aware
/// shape used by `execute_tokio_future*`).
pub fn spawn<F>(fut: F) -> JoinHandle<()>
where
  F: 'static + Send + Future<Output = ()>,
{
  JoinHandle::spawn(fut)
}

#[cfg(all(feature = "runtime_executor", not(feature = "noop")))]
/// Run a future to completion on the calling thread. Uses `pollster::block_on`,
/// which is reentrant — nested `block_on` calls don't panic.
pub fn block_on<F: Future>(fut: F) -> F::Output {
  pollster::block_on(fut)
}

#[cfg(all(feature = "runtime_executor", not(feature = "noop")))]
/// Run a blocking closure on a dedicated OS thread. Returns a `JoinHandle`
/// whose `.await` yields the closure's return value.
///
/// Unlike tokio's pooled `spawn_blocking`, this spawns a fresh thread per
/// call. Suitable for the napi-internal use cases (a small number of
/// long-running blocking tasks); not intended as a general-purpose blocking
/// pool.
pub fn spawn_blocking<F, R>(func: F) -> JoinHandle<R>
where
  F: FnOnce() -> R + Send + 'static,
  R: Send + 'static,
{
  // Drive the closure on a dedicated thread, then forward its result
  // through the standard JoinHandle (which spawns onto the ThreadPool).
  // Two indirections, but it keeps the public shape consistent.
  JoinHandle::spawn(async move {
    let (tx, rx) = futures::channel::oneshot::channel();
    std::thread::spawn(move || {
      let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(func));
      let _ = tx.send(outcome);
    });
    match rx.await {
      Ok(Ok(v)) => v,
      Ok(Err(panic)) => std::panic::resume_unwind(panic),
      Err(_) => panic!("napi spawn_blocking task dropped before completion"),
    }
  })
}

#[cfg(all(feature = "runtime_executor", not(feature = "noop")))]
/// No-op for the executor backend — there's no per-call runtime context to
/// enter. Provided to keep the public surface in sync with the tokio_rt
/// implementation.
pub fn within_runtime_if_available<F: FnOnce() -> T, T>(f: F) -> T {
  f()
}

#[cfg(all(feature = "runtime_executor", not(feature = "noop")))]
/// No-op for the executor backend; the ThreadPool initializes lazily.
pub fn start_async_runtime() {}

#[cfg(all(feature = "runtime_executor", not(feature = "noop")))]
/// No-op for the executor backend; the ThreadPool's threads exit when the
/// process does. Provided to match the tokio_rt API surface.
pub fn shutdown_async_runtime() {}

// ---------------------------------------------------------------------------
// `noop` backend
// ---------------------------------------------------------------------------

#[cfg(feature = "noop")]
/// Runs a future to completion
/// This is blocking, meaning that it pauses other execution until the future is complete,
/// only use it when it is absolutely necessary, in other places use async functions instead.
pub fn block_on<F: Future>(_: F) -> F::Output {
  unreachable!("noop feature is enabled, block_on is not available")
}

#[cfg(feature = "noop")]
pub fn within_runtime_if_available<F: FnOnce() -> T, T>(f: F) -> T {
  f()
}

// ---------------------------------------------------------------------------
// `execute_tokio_future*` — public façade used by the `#[napi] async fn`
// macro. Body switches on backend; signature is identical.
// ---------------------------------------------------------------------------

#[cfg(feature = "noop")]
#[allow(unused)]
pub fn execute_tokio_future<
  Data: 'static + Send,
  Fut: 'static + Send + Future<Output = std::result::Result<Data, impl Into<Error>>>,
  Resolver: 'static + FnOnce(sys::napi_env, Data) -> Result<sys::napi_value>,
>(
  env: sys::napi_env,
  fut: Fut,
  resolver: Resolver,
) -> Result<sys::napi_value> {
  Ok(std::ptr::null_mut())
}

#[cfg(not(feature = "noop"))]
#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub fn execute_tokio_future<
  Data: 'static + Send,
  Fut: 'static + Send + Future<Output = std::result::Result<Data, impl Into<Error>>>,
  Resolver: 'static + FnOnce(sys::napi_env, Data) -> Result<sys::napi_value>,
>(
  env: sys::napi_env,
  fut: Fut,
  resolver: Resolver,
) -> Result<sys::napi_value> {
  let env = Env::from_raw(env);
  let (deferred, promise) = JsDeferred::new(&env)?;
  #[cfg(any(
    all(target_family = "wasm", tokio_unstable),
    not(target_family = "wasm")
  ))]
  let deferred_for_panic = deferred.clone();
  let sendable_resolver = SendableResolver::new(resolver);

  let inner = async move {
    match fut.await {
      Ok(v) => deferred.resolve(move |env| {
        sendable_resolver
          .resolve(env.raw(), v)
          .map(|v| unsafe { Unknown::from_raw_unchecked(env.raw(), v) })
      }),
      Err(e) => deferred.reject(e.into()),
    }
  };

  #[cfg(any(
    all(target_family = "wasm", tokio_unstable),
    not(target_family = "wasm")
  ))]
  let jh = spawn(inner);

  #[cfg(any(
    all(target_family = "wasm", tokio_unstable),
    not(target_family = "wasm")
  ))]
  spawn(async move {
    if let Err(err) = jh.await {
      if let Ok(reason) = err.try_into_panic() {
        if let Some(s) = reason.downcast_ref::<&str>() {
          deferred_for_panic.reject(Error::new(crate::Status::GenericFailure, s));
        } else {
          deferred_for_panic.reject(Error::new(
            crate::Status::GenericFailure,
            "Panic in async function",
          ));
        }
      }
    }
  });

  #[cfg(all(target_family = "wasm", not(tokio_unstable)))]
  {
    std::thread::spawn(|| {
      block_on(inner);
    });
  }

  Ok(promise.0.value)
}

#[doc(hidden)]
#[cfg(not(feature = "noop"))]
#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub fn execute_tokio_future_with_finalize_callback<
  Data: 'static + Send,
  Fut: 'static + Send + Future<Output = std::result::Result<Data, impl Into<Error>>>,
  Resolver: 'static + FnOnce(sys::napi_env, Data) -> Result<sys::napi_value>,
>(
  env: sys::napi_env,
  fut: Fut,
  resolver: Resolver,
  finalize_callback: Option<Box<dyn FnOnce(sys::napi_env)>>,
) -> Result<sys::napi_value> {
  let env = Env::from_raw(env);
  let (mut deferred, promise) = JsDeferred::new(&env)?;
  deferred.set_finalize_callback(finalize_callback);
  #[cfg(any(
    all(target_family = "wasm", tokio_unstable),
    not(target_family = "wasm")
  ))]
  let deferred_for_panic = deferred.clone();
  let sendable_resolver = SendableResolver::new(resolver);

  let inner = async move {
    match fut.await {
      Ok(v) => deferred.resolve(move |env| {
        sendable_resolver
          .resolve(env.raw(), v)
          .map(|v| unsafe { Unknown::from_raw_unchecked(env.raw(), v) })
      }),
      Err(e) => deferred.reject(e.into()),
    }
  };

  #[cfg(any(
    all(target_family = "wasm", tokio_unstable),
    not(target_family = "wasm")
  ))]
  let jh = spawn(inner);

  #[cfg(any(
    all(target_family = "wasm", tokio_unstable),
    not(target_family = "wasm")
  ))]
  spawn(async move {
    if let Err(err) = jh.await {
      if let Ok(reason) = err.try_into_panic() {
        if let Some(s) = reason.downcast_ref::<&str>() {
          deferred_for_panic.reject(Error::new(crate::Status::GenericFailure, s));
        } else {
          deferred_for_panic.reject(Error::new(
            crate::Status::GenericFailure,
            "Panic in async function",
          ));
        }
      }
    }
  });

  #[cfg(all(target_family = "wasm", not(tokio_unstable)))]
  {
    std::thread::spawn(|| {
      block_on(inner);
    });
  }

  Ok(promise.0.value)
}

#[cfg(feature = "noop")]
#[doc(hidden)]
pub fn execute_tokio_future_with_finalize_callback<
  Data: 'static + Send,
  Fut: 'static + Send + Future<Output = std::result::Result<Data, impl Into<Error>>>,
  Resolver: 'static + FnOnce(sys::napi_env, Data) -> Result<sys::napi_value>,
>(
  _env: sys::napi_env,
  _fut: Fut,
  _resolver: Resolver,
  _finalize_callback: Option<Box<dyn FnOnce(sys::napi_env)>>,
) -> Result<sys::napi_value> {
  Ok(std::ptr::null_mut())
}

pub struct AsyncBlockBuilder<
  V: Send + 'static,
  F: Future<Output = Result<V>> + Send + 'static,
  Dispose: FnOnce(Env) -> Result<()> + 'static = fn(Env) -> Result<()>,
> {
  inner: F,
  dispose: Option<Dispose>,
}

impl<V: ToNapiValue + Send + 'static, F: Future<Output = Result<V>> + Send + 'static>
  AsyncBlockBuilder<V, F>
{
  /// Create a new `AsyncBlockBuilder` with the given future, without dispose
  pub fn new(inner: F) -> Self {
    Self {
      inner,
      dispose: None,
    }
  }
}

impl<
    V: ToNapiValue + Send + 'static,
    F: Future<Output = Result<V>> + Send + 'static,
    Dispose: FnOnce(Env) -> Result<()> + 'static,
  > AsyncBlockBuilder<V, F, Dispose>
{
  pub fn with(inner: F) -> Self {
    Self {
      inner,
      dispose: None,
    }
  }

  pub fn with_dispose(mut self, dispose: Dispose) -> Self {
    self.dispose = Some(dispose);
    self
  }

  pub fn build(self, env: &Env) -> Result<AsyncBlock<V>> {
    Ok(AsyncBlock {
      inner: execute_tokio_future(env.0, self.inner, |env, v| unsafe {
        if let Some(dispose) = self.dispose {
          let env = Env::from_raw(env);
          dispose(env)?;
        }
        V::to_napi_value(env, v)
      })?,
      _phantom: PhantomData,
    })
  }
}

impl<V: Send + 'static, F: Future<Output = Result<V>> + Send + 'static> AsyncBlockBuilder<V, F> {
  /// Create a new `AsyncBlockBuilder` with the given future, without dispose
  pub fn build_with_map<T: ToNapiValue, Map: FnOnce(Env, V) -> Result<T> + 'static>(
    env: &Env,
    inner: F,
    map: Map,
  ) -> Result<AsyncBlock<T>> {
    Ok(AsyncBlock {
      inner: execute_tokio_future(env.0, inner, |env, v| unsafe {
        let v = map(Env::from_raw(env), v)?;
        T::to_napi_value(env, v)
      })?,
      _phantom: PhantomData,
    })
  }
}

pub struct AsyncBlock<T: ToNapiValue + 'static> {
  inner: sys::napi_value,
  _phantom: PhantomData<T>,
}

impl<T: ToNapiValue + 'static> ToNapiValue for AsyncBlock<T> {
  unsafe fn to_napi_value(_: napi_sys::napi_env, val: Self) -> Result<napi_sys::napi_value> {
    Ok(val.inner)
  }
}
