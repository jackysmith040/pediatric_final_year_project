import pandas as pd

df = pd.read_csv('runs/child_room_drawers/per_frame.csv')
for tid in [1, 2, 3, 9, 12]:
    sub = df[df['track_id'] == tid]
    if len(sub) > 0:
        f_min = sub['frame'].min()
        f_max = sub['frame'].max()
        b = [int(sub['box_x1'].mean()), int(sub['box_y1'].mean()), int(sub['box_x2'].mean()), int(sub['box_y2'].mean())]
        print(f"Track #{tid}: frames {f_min}-{f_max} (count={len(sub)}), mean_box={b}")
