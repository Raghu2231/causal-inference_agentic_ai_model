export default function KpiCard({ label, value }) {
  return (
    <div className="metric">
      <div style={{ fontSize: 13, opacity: 0.75 }}>{label}</div>
      <div style={{ fontSize: 22, fontWeight: 700 }}>{value}</div>
    </div>
  );
}
