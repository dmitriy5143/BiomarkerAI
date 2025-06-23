from app import create_app
from celery import Celery

def make_celery(app):
    celery = Celery(
        app.import_name,
        broker=app.config.get("CELERY_BROKER_URL", "redis://localhost:6379/0"),
        backend=app.config.get("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")
    )
    celery.conf.update(app.config)

    class ContextTask(celery.Task):
        def __call__(self, *args, **kwargs):
            with app.app_context():
                return self.run(*args, **kwargs)
    celery.Task = ContextTask
    return celery

flask_app = create_app()
celery = make_celery(flask_app)
celery.autodiscover_tasks(['tasks'])