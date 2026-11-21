import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Flight Fare Predictor",
  description: "Interactive machine learning flight price prediction demo",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
