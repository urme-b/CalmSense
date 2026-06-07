// Client-side CSV parsing for the prediction panel. The dashboard runs the
// model in-browser (see services/onnx.ts), so no backend calls are needed.

export const parseCSV = async (file: File): Promise<Record<string, number>[]> => {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const text = (e.target?.result as string) || '';
        const lines = text.trim().split('\n').filter((l) => l.trim().length > 0);
        if (lines.length < 2) {
          throw new Error('CSV needs a header row and at least one data row');
        }
        const headers = lines[0].split(',').map((h) => h.trim());
        const data: Record<string, number>[] = [];
        for (let i = 1; i < lines.length; i++) {
          const values = lines[i].split(',');
          const row: Record<string, number> = {};
          headers.forEach((header, index) => {
            row[header] = parseFloat((values[index] ?? '').trim());
          });
          data.push(row);
        }
        resolve(data);
      } catch (error) {
        reject(error);
      }
    };
    reader.onerror = reject;
    reader.readAsText(file);
  });
};
