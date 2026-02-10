export default function UploadPage({ onUpload, loading }) {
  const handleChange = (event) => {
    if (!event.target.files?.length) {
      return;
    }
    onUpload(event.target.files[0]);
  };

  return (
    <div className="card">
      <h2>Upload Page</h2>
      <p>Select a .xlsx or .xls file to auto-detect schema and start the workflow.</p>
      <input type="file" accept=".xlsx,.xls" onChange={handleChange} disabled={loading} />
    </div>
  );
}
