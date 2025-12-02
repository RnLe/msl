import './layout.css';
import { Inter } from 'next/font/google';

const inter = Inter({
  subsets: ['latin'],
  weight: ['100', '200', '300', '400', '500', '600', '700'],
  display: 'swap',
});

export const metadata = {
  title: '2D MPB | MSL Framework',
  description: 'Photonic band diagram visualization',
};

export default function MPB2DLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <div className={`mpb2d-layout ${inter.className}`}>
      {children}
    </div>
  );
}
