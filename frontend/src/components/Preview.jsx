export default function Preview({ file }) {
  return (
    <div style={{ marginTop: 20 }}>
      <img src={file} alt="preview" width="300" />
    </div>
  );
}