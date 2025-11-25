"use client";
import React, { useState } from "react";
import Sidebar from '@/components/sidebar/Sidebar'
import Link from 'next/link';
import Image from 'next/image';
import LOGO from '@/assets/Cloud_Pulse.png';
import { Menu } from 'lucide-react';

type Props = {
  children: React.ReactNode;
};

const Layout = ({ children }: Props) => {
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);

  const handleToggleSidebar = () => setIsSidebarOpen((open) => !open);

  return (
    <div className="flex flex-col h-screen overflow-hidden bg-[#F9FEFF]">
      <header className="w-full h-[72px] bg-[#F9FEFF] flex items-center pl-5 pr-5 shadow-sm z-50 sticky top-0">
        <button
          onClick={handleToggleSidebar}
          className="mr-4 p-2 rounded-md bg-[#233E7D] text-white hover:bg-[#1a2d5c] transition-colors"
          aria-label={isSidebarOpen ? 'Close sidebar' : 'Open sidebar'}
        >
          <Menu size={28} />
        </button>
        <Link href="/landingpage">
          <Image src={LOGO} alt="Cloud Pulse" width={115} height={54} priority />
        </Link>
      </header>
      <div className="flex-1 overflow-auto flex flex-row">
        <Sidebar isOpen={isSidebarOpen} setIsOpen={setIsSidebarOpen} />
        <main
          className={`flex-1 overflow-auto transition-all duration-300 ease-in-out pl-[22px] ${
            isSidebarOpen ? "ml-[80px]" : "ml-0"
          } bg-[#F9FEFF]`}
        >
          {/* Example: Add a wrapper for card content */}
          <div className="max-w-7xl mx-auto py-8">
            {children}
          </div>
        </main>
      </div>
    </div>
  );
};

export default Layout;