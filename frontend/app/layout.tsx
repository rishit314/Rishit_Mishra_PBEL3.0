import React from 'react';
import './globals.css';
export const metadata = {
  title: 'AgroVision AI - Crop Disease Diagnosis',
  description: 'Production RAG & Computer Vision platform for plant pathology.',
};

// This default export is required by Next.js App Router!
// If this function is missing or exported incorrectly, ALL routes (/, /404) crash immediately.
export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className="bg-slate-950 text-slate-100 antialiased min-h-screen">
        {children}
      </body>
    </html>
  );
}