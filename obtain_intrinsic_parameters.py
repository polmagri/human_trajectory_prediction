import pyrealsense2 as rs

# Initialize the RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()

# Configure the RGB and Depth streams
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)

# Start the pipeline
profile = pipeline.start(config)

# Get the intrinsic parameters of the RGB sensor
color_intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

print("Width: ", color_intrinsics.width)
print("Height: ", color_intrinsics.height)
print("Principal point (ppx, ppy): ", color_intrinsics.ppx, color_intrinsics.ppy)
print("Focal length (fx, fy): ", color_intrinsics.fx, color_intrinsics.fy)
print("Distortion model: ", color_intrinsics.model)
print("Distortion coefficients: ", color_intrinsics.coeffs)

# Stop the pipeline
pipeline.stop()