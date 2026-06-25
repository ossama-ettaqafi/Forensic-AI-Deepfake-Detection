import { useState, useRef } from "react";

export default function UploadBox({ onFileSelect, loading, progress }) {
  const [preview, setPreview] = useState(null);
  const [type, setType] = useState(null);
  const inputRef = useRef(null);

  const handleFile = (file) => {
    if (!file) return;

    const isImage = file.type.startsWith("image/");
    const isVideo = file.type.startsWith("video/");

    if (!isImage && !isVideo) {
      alert("Only image or video allowed");
      return;
    }

    setType(isImage ? "image" : "video");

    const url = URL.createObjectURL(file);
    setPreview(url);

    onFileSelect(file);
  };

  const onDrop = (e) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    handleFile(file);
  };

  const onDragOver = (e) => e.preventDefault();

  return (
    <div
      className="card p-6 border-2 border-dashed rounded-xl text-center"
      onDrop={onDrop}
      onDragOver={onDragOver}
    >
      {/* Upload button */}
      <input
        ref={inputRef}
        type="file"
        accept="image/*,video/*"
        hidden
        onChange={(e) => handleFile(e.target.files[0])}
      />

      <button
        onClick={() => inputRef.current.click()}
        className="px-4 py-2 bg-black text-white rounded"
      >
        Upload Image / Video
      </button>

      <p className="text-sm text-gray-500 mt-2">
        or drag & drop here
      </p>

      {/* Preview */}
      {preview && type === "image" && (
        <img
          src={preview}
          className="mt-4 max-h-64 mx-auto rounded"
        />
      )}

      {preview && type === "video" && (
        <video
          src={preview}
          controls
          className="mt-4 max-h-64 mx-auto rounded"
        />
      )}

      {/* Loading state */}
      {loading && (
        <div className="mt-4">
          <div className="spinner" />
          <p className="mt-2 text-sm">
            {progress?.text || "Analyzing..."}
          </p>

          {/* Progress bar */}
          {progress?.value !== undefined && (
            <div className="w-full bg-gray-200 h-2 rounded mt-2">
              <div
                className="bg-blue-500 h-2 rounded"
                style={{ width: `${progress.value}%` }}
              />
            </div>
          )}
        </div>
      )}
    </div>
  );
}