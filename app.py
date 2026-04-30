import streamlit as st
import cv2
import numpy as np
import io
from PIL import Image

st.title("Color + Frequency Image Processing App")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:

    # ✅ Improved file size handling
    if uploaded_file.size > 15 * 1024 * 1024:
        st.error("File too large. Max allowed is 15MB")
        st.stop()

    if uploaded_file.size > 5 * 1024 * 1024:
        st.warning("Large image detected. It will be resized for performance.")

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)

    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is None:
        st.error("Error loading image. Please upload a valid image file.")
        st.stop()

    # ✅ Resize large images (prevents memory crash)
    max_size = 512
    h, w = image.shape[:2]

    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        image = cv2.resize(image, (int(w * scale), int(h * scale)))

    # Convert to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    st.subheader("Original Image")
    st.image(image, use_column_width=True)

    # ---------------- COLOR PROCESSING ----------------
    st.subheader("Color Processing")

    r = st.slider("Red Channel", 0, 200, 100)
    g = st.slider("Green Channel", 0, 200, 100)
    b = st.slider("Blue Channel", 0, 200, 100)

    color_img = image.copy().astype(np.float32)
    color_img[:, :, 0] *= (r / 100)
    color_img[:, :, 1] *= (g / 100)
    color_img[:, :, 2] *= (b / 100)
    color_img = np.clip(color_img, 0, 255).astype(np.uint8)

    st.image(color_img, caption="RGB Adjusted", use_column_width=True)

    # Grayscale
    if st.checkbox("Convert to Grayscale"):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        st.image(gray, caption="Grayscale", use_column_width=True)
    else:
        gray = cv2.cvtColor(color_img, cv2.COLOR_RGB2GRAY)

    # ---------------- FREQUENCY DOMAIN ----------------
    st.subheader("Frequency Domain Processing")

    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)

    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
    magnitude_spectrum = cv2.normalize(
        magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)

    st.image(magnitude_spectrum, caption="Magnitude Spectrum", use_column_width=True)

    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2

    filter_type = st.selectbox("Filter Type", ["Low Pass", "High Pass"])
    radius = st.slider("Filter Radius", 5, 100, 20)

    # Mask
    x = np.arange(rows)
    y = np.arange(cols)
    X, Y = np.meshgrid(x, y, indexing='ij')

    dist = (X - crow)**2 + (Y - ccol)**2
    mask = np.zeros((rows, cols), np.uint8)

    if filter_type == "Low Pass":
        mask[dist <= radius**2] = 1
    else:
        mask[dist > radius**2] = 1

    st.image(mask * 255, caption="Filter Mask", use_column_width=True)

    # Apply filter
    filtered = fshift * mask

    filtered_spectrum = 20 * np.log(np.abs(filtered) + 1)
    filtered_spectrum = cv2.normalize(
        filtered_spectrum, None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)

    st.image(filtered_spectrum, caption="Filtered Spectrum", use_column_width=True)

    # Inverse FFT
    f_ishift = np.fft.ifftshift(filtered)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)

    img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    st.image(img_back, caption="Filtered Image", use_column_width=True)

    # ✅ Proper download button
    st.subheader("Download Result")

    img_pil = Image.fromarray(img_back)
    buf = io.BytesIO()
    img_pil.save(buf, format="PNG")

    st.download_button(
        label="Download Image",
        data=buf.getvalue(),
        file_name="processed.png",
        mime="image/png"
    )