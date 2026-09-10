from pediatric_counter.app.config import load_config
from pediatric_counter.app.pipeline import run_pipeline

cfg = load_config('pediatric_counter/configs/drawers.yaml')
cfg.artifacts.live_view = False
cfg.artifacts.save_annotated_video = False
cfg.room.max_frames = 1057

summary = run_pipeline(cfg)
print('='*50)
print('CHILD ROOM DRAWERS VERIFICATION RESULT:')
print(f'Children: {summary.distinct_child_count}, Adults: {summary.distinct_adult_count}, Total: {summary.total_distinct_count}')
print(f'Child IDs: {summary.counted_child_ids}')
print(f'Adult IDs: {summary.counted_adult_ids}')
print('='*50)
