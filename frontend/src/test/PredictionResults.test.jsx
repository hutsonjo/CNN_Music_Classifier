import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import PredictionResults from '../components/PredictionResults.jsx';

const sample = [
  { genre: 'rock', confidence: 0.55 },
  { genre: 'metal', confidence: 0.25 },
  { genre: 'jazz', confidence: 0.10 },
  { genre: 'pop', confidence: 0.10 },
];

describe('PredictionResults', () => {
  it('renders nothing when given an empty prediction list', () => {
    const { container } = render(<PredictionResults predictions={[]} />);
    expect(container.firstChild).toBeNull();
  });

  it('renders nothing when given null', () => {
    const { container } = render(<PredictionResults predictions={null} />);
    expect(container.firstChild).toBeNull();
  });

  
  // Ranking / display
  

  it('displays the top prediction prominently with confidence percentage', () => {
    render(<PredictionResults predictions={sample} />);

    expect(screen.getByRole('heading', { name: /rock/i })).toBeInTheDocument();
    expect(screen.getByText(/55\.0% confidence/)).toBeInTheDocument();
  });

  it('renders one bar per prediction in descending confidence order', () => {
    render(<PredictionResults predictions={sample} />);

    const meters = screen.getAllByRole('meter');
    expect(meters).toHaveLength(4);
    expect(meters[0]).toHaveAttribute('aria-label', 'rock confidence');
    expect(meters[1]).toHaveAttribute('aria-label', 'metal confidence');
  });

  it('sorts predictions even when the input is out of order', () => {
    const shuffled = [
      { genre: 'jazz', confidence: 0.10 },
      { genre: 'rock', confidence: 0.55 },
      { genre: 'metal', confidence: 0.25 },
    ];
    render(<PredictionResults predictions={shuffled} />);

    expect(screen.getByRole('heading', { name: /rock/i })).toBeInTheDocument();
  });

  
  // Bar rendering edge cases

  it('clamps confidence above 1 to 100%', () => {
    render(
      <PredictionResults
        predictions={[{ genre: 'rock', confidence: 1.5 }]}
      />
    );
    const meter = screen.getByRole('meter');
    expect(meter).toHaveAttribute('aria-valuenow', '100');
  });

  it('clamps negative confidence to 0%', () => {
    render(
      <PredictionResults
        predictions={[{ genre: 'rock', confidence: -0.2 }]}
      />
    );
    const meter = screen.getByRole('meter');
    expect(meter).toHaveAttribute('aria-valuenow', '0');
  });

  it('shows the exact percentage value for each genre', () => {
    render(<PredictionResults predictions={sample} />);

    expect(screen.getByText('25.0%')).toBeInTheDocument();
    expect(screen.getAllByText('10.0%')).toHaveLength(2);
  });
});