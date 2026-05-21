/**
 * Ranked genre confidence display.
 *
 * Renders a horizontal bar per genre, sorted by confidence (descending).
 * The top prediction is highlighted as the "winning" guess. The bars
 * grow from 0–100% and the numeric value sits at the right edge so the
 * user can read both shape and exact score at a glance.
 *
 * Input is the array returned by the API:
 *   [{ genre: "rock", confidence: 0.721 }, ...]
 */
export default function PredictionResults({ predictions }) {
  if (!predictions?.length) return null;

  // Defensive copy + sort: don't mutate parent state.
  const ranked = [...predictions].sort((a, b) => b.confidence - a.confidence);
  const top = ranked[0];

  return (
    <section className="results" aria-live="polite">
      <div className="results__headline">
        <span className="results__label">Predicted genre</span>
        <h2 className="results__top">{top.genre}</h2>
        <span className="results__confidence">
          {(top.confidence * 100).toFixed(1)}% confidence
        </span>
      </div>

      <ul className="bar-list" aria-label="All genre confidence scores">
        {ranked.map((p, idx) => (
          <ConfidenceBar
            key={p.genre}
            genre={p.genre}
            confidence={p.confidence}
            isTop={idx === 0}
          />
        ))}
      </ul>
    </section>
  );
}

function ConfidenceBar({ genre, confidence, isTop }) {
  const pct = Math.max(0, Math.min(1, confidence)) * 100;
  return (
    <li className={`bar ${isTop ? 'bar--top' : ''}`}>
      <span className="bar__label">{genre}</span>
      <div
        className="bar__track"
        role="meter"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={Number(pct.toFixed(1))}
        aria-label={`${genre} confidence`}
      >
        <div className="bar__fill" style={{ width: `${pct}%` }} />
      </div>
      <span className="bar__value">{pct.toFixed(1)}%</span>
    </li>
  );
}