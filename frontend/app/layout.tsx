import React from 'react';
import "./globals.css";
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
    <html lang="en">
      <body className="bg-[#DFE8E1] text-[#182420] antialiased min-h-screen">
        {children}
      </body>
    </html>
  );
}