from celery_app import celery
from pipeline.main import execute_pipeline

@celery.task(bind=True)
def process_pipeline_task(self, params):
    def progress_update(meta):
        self.update_state(state="PROGRESS", meta=meta)
    
    def llm_progress_update(result, index):
        payload = {"phase": "llm_analysis", "result": dict(result, index=index)}
        self.update_state(state="PROGRESS", meta=payload)
    
    result, _ = execute_pipeline(params, update_progress_fn=progress_update, llm_progress_fn=llm_progress_update)
    return result