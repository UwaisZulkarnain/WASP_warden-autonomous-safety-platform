from ultralytics import YOLO

model = YOLO(
    r"C:\Users\User\OneDrive\STUDY\UTM KL\UTM\Project\AI_Showcase\runs\detect\wasp_ppe_v1\weights\best.pt"
)

results = model.predict(
    source=0,
    conf=0.5,
    show=True
)