"use client";

import React from "react";

type Props = { children: React.ReactNode };

const Layout = ({ children }: Props) => {
  return (
    <div className="flex flex-col min-h-screen bg-[#F9FEFF]">
      <div className="flex-grow ">
        {children}
      </div>
    </div>
  );
};

export default Layout;
