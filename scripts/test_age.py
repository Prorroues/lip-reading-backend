from age_estimation.fastapi_age import AgeEstimator
import os
from Visual_Speech_Recognition_for_Multiple_Languages.fastapi_vsr import VSRInference
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'Visual_Speech_Recognition_for_Multiple_Languages')))

var = VSRInference()
text = var.process_videos("uploads/videos/2.mp4")
# import asyncio
# from utils.rotate_videos import rotate_video_sync
# from concurrent.futures import ProcessPoolExecutor
#
# executor = ProcessPoolExecutor()
#
#
# async def main():
#     loop = asyncio.get_running_loop()
#     await loop.run_in_executor(executor, rotate_video_sync, "uploads/videos/2025349949.mp4", "uploads/videos/2.mp4")
#
# asyncio.run(main())
# video_path = "uploads/videos/2025349949.mp4"
# output_path = "uploads/videos/2.mp4"
# asyncio.run(convert_video(video_path, output_path))

# model = AgeEstimator(face_size=64, weights="age_estimation/weights/weights.pt", device="cpu")
# model.prediction_videos("uploads/videos/2.mp4", "generates/images/2.png")
# # predicted_image = model.predict("age_estimation/Images/trang.jpg")
# save_dir = os.path.join("runs", "predict")
# os.makedirs(save_dir, exist_ok=True)
# save_path = os.path.join(save_dir, "results.jpg")
# plt.imsave(save_path, predicted_image)
# print(f"Result saved to {save_path}")