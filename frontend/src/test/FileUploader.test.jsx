import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import FileUploader from '../components/FileUploader.jsx';

// Helper: create a fake file with a specific size in MB.
function makeFile(name, sizeMB = 0.1) {
  const content = new Uint8Array(Math.max(1, Math.floor(sizeMB * 1024 * 1024)));
  return new File([content], name, { type: 'audio/wav' });
}


// File selection

describe('FileUploader', () => {
  it('displays the selected file name and enables the submit button', async () => {
    const user = userEvent.setup();
    render(<FileUploader onSubmit={vi.fn()} isSubmitting={false} />);

    const input = screen.getByTestId('file-input');
    const file = makeFile('mysong.wav');
    await user.upload(input, file);

    expect(screen.getByText('mysong.wav')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /classify genre/i })).toBeEnabled();
  });

  it('disables the submit button when no file is selected', () => {
    render(<FileUploader onSubmit={vi.fn()} isSubmitting={false} />);
    expect(screen.getByRole('button', { name: /classify genre/i })).toBeDisabled();
  });

  it('disables the submit button while a request is in flight', async () => {
    const user = userEvent.setup();
    const { rerender } = render(
      <FileUploader onSubmit={vi.fn()} isSubmitting={false} />
    );
    await user.upload(screen.getByTestId('file-input'), makeFile('a.wav'));

    rerender(<FileUploader onSubmit={vi.fn()} isSubmitting={true} />);
    expect(screen.getByRole('button', { name: /classifying/i })).toBeDisabled();
  });

  
  // Validation — wrong extension
  
  it('rejects unsupported file types with an inline error', () => {
    render(<FileUploader onSubmit={vi.fn()} isSubmitting={false} />);

    // Bypass the `accept` attribute by firing the change event directly —
    // a user could still drag a .pdf onto the dropzone, so we want the
    // component to reject it regardless of how it got in.
    const input = screen.getByTestId('file-input');
    const badFile = new File(['x'], 'document.pdf', { type: 'application/pdf' });
    fireEvent.change(input, { target: { files: [badFile] } });

    expect(screen.getByRole('alert')).toHaveTextContent(/unsupported file type/i);
    expect(screen.getByRole('button', { name: /classify genre/i })).toBeDisabled();
  });

  
  // Validation — file too large
  
  it('rejects files larger than the size limit', async () => {
    const user = userEvent.setup();
    render(<FileUploader onSubmit={vi.fn()} isSubmitting={false} />);

    // 31 MB > 30 MB cap
    const big = makeFile('huge.wav', 31);
    await user.upload(screen.getByTestId('file-input'), big);

    expect(screen.getByRole('alert')).toHaveTextContent(/too large/i);
  });

  
  // Submission

  it('invokes onSubmit with the selected file when the button is clicked', async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn();
    render(<FileUploader onSubmit={onSubmit} isSubmitting={false} />);

    const file = makeFile('jazz.wav');
    await user.upload(screen.getByTestId('file-input'), file);
    await user.click(screen.getByRole('button', { name: /classify genre/i }));

    expect(onSubmit).toHaveBeenCalledTimes(1);
    expect(onSubmit).toHaveBeenCalledWith(file);
  });

  it('does not call onSubmit while a previous request is still in flight', async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn();
    const { rerender } = render(
      <FileUploader onSubmit={onSubmit} isSubmitting={false} />
    );
    await user.upload(screen.getByTestId('file-input'), makeFile('a.wav'));

    rerender(<FileUploader onSubmit={onSubmit} isSubmitting={true} />);
    // Button is disabled in this state, but try anyway:
    const btn = screen.getByRole('button', { name: /classifying/i });
    await user.click(btn);

    expect(onSubmit).not.toHaveBeenCalled();
  });
});