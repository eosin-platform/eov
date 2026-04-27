use std::any::Any;
use std::sync::OnceLock;
use std::sync::mpsc;

type UiJob = Box<dyn FnOnce() + Send + 'static>;

fn slint_ui_test_sender() -> &'static mpsc::Sender<UiJob> {
    static SENDER: OnceLock<mpsc::Sender<UiJob>> = OnceLock::new();
    SENDER.get_or_init(|| {
        let (tx, rx) = mpsc::channel::<UiJob>();
        std::thread::Builder::new()
            .name("slint-ui-test-thread".to_string())
            .spawn(move || {
                i_slint_backend_testing::init_integration_test_with_mock_time();
                for job in rx {
                    job();
                }
            })
            .expect("failed to spawn Slint UI test thread");
        tx
    })
}

pub(crate) fn run_on_slint_ui_test_thread<R>(f: impl FnOnce() -> R + Send + 'static) -> R
where
    R: Send + 'static,
{
    let (tx, rx) = mpsc::sync_channel::<std::thread::Result<R>>(1);
    let job = Box::new(move || {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f));
        let _ = tx.send(result);
    });
    slint_ui_test_sender()
        .send(job)
        .expect("failed to enqueue Slint UI test job");
    match rx.recv().expect("failed to receive Slint UI test result") {
        Ok(result) => result,
        Err(payload) => std::panic::resume_unwind(payload as Box<dyn Any + Send>),
    }
}
