import streamlit as st
import numpy as np
from streamlit_drawable_canvas import st_canvas
from PIL import Image  # Correct import
import utils

# Canvas settings
stroke_width = 10

st.sidebar.title("Draw a Digit")
canvas_result = st_canvas(
    fill_color="black",  # Black background
    stroke_width=stroke_width,
    stroke_color="white",  # White drawing
    background_color="black",  # Black canvas
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas",
)

# If something is drawn on the canvas
if canvas_result.image_data is not None:
    # Convert the drawn image to grayscale
    user_image = np.array(canvas_result.image_data[:, :, :3])
    
    # Convert the image to grayscale (optional, since it's black and white already)
    user_image = np.dot(user_image, [0.2989, 0.587, 0.114])
    
    # Resize to 28x28 (MNIST format)
    user_image = np.array(Image.fromarray(user_image).resize((28, 28)))

    # Invert colors: background black -> 0, drawing white -> 255
    user_image = 255 - user_image

    # Ensure the image data is in the correct uint8 format
    user_image = user_image.astype(np.uint8)

    st.sidebar.image(user_image, caption="Your Digit", use_column_width=True)

    # Find the nearest 5 MNIST images
    nearest_images = utils.find_nearest_images(user_image, utils.load_mnist_images())

    # Display the nearest images on the right
    st.title("Nearest MNIST Images")
    cols = st.columns(5)
    for i, (image, distance) in enumerate(nearest_images):
        with cols[i]:
            st.image(image, caption=f"Distance: {distance:.2f}", use_column_width=True)
