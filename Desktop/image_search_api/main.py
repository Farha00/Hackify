from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from model import processor, model, device, index, doc_paths
from utils import preprocess_image, extract_feature, cosine_similarity
import numpy as np
import traceback

app = FastAPI()

@app.post("/search")
async def search_image(file: UploadFile = File(...)):
    try:
        print("✅ Step 1: File received:", file.filename)

        image_data = await file.read()

        print("✅ Step 2: Preprocessing image...")
        pixel_values = preprocess_image(image_data, processor)
        if pixel_values is None:
            raise ValueError("❌ Image preprocessing failed.")

        print("✅ Step 3: Extracting feature...")
        feature = extract_feature(pixel_values, model, device)

        print("✅ Step 4: FAISS search...")
        k = 3
        D, I = index.search(np.expand_dims(feature, axis=0), k)

        print("✅ Step 5: Fetching top paths...")
        top_paths = [doc_paths[i] for i in I[0]]

        print("✅ Step 6: Calculating similarities...")
        top_features = np.take(index.reconstruct_n(0, index.ntotal), I[0], axis=0)
        sims = cosine_similarity(feature, top_features)

        results = [
            {
                "rank": i + 1,
                "path": top_paths[i],
                "cosine_similarity": float(sims[i])
            }
            for i in range(k)
        ]

        return JSONResponse(content={"results": results})

    except Exception as e:
        print("❌ Error during inference:")
        traceback.print_exc()
        return JSONResponse(content={"error": str(e)}, status_code=500)

