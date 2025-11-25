"use client";
import React from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";

type MenuItemProps = {
  href: string;
  label: string;
  isActive: boolean;
};

const MenuItem: React.FC<MenuItemProps> = ({ href, label, isActive }) => {
  return (
    <Link href={href} passHref>
      <div
        className={`
          py-2 px-4 mb-2 rounded-md transition-colors duration-200 text-base font-semibold cursor-pointer
          ${
            isActive
              ? "bg-[#233E7D] text-white border-l-4 border-[#D82026]"
              : "text-[#233E7D] hover:text-[#D82026] hover:bg-[#233E7D]/10"
          }
        `}
      >
        {label}
      </div>
    </Link>
  );
};

type Props = {
  children: React.ReactNode;
};

const Layout = ({ children }: Props) => {
  const pathname = usePathname();

  const menuItems = [
    { href: "/alerts/alertRules", label: "Alert Rules" },
    { href: "/alerts/contactPoints", label: "Contact Points" },
  ];

  return (
    <div className="flex overflow-hidden h-screen bg-[#F9FEFF]">
      <div className="h-full w-[15%] p-4 bg-[#F9FEFF] border-r border-[#E0E5EF]">
        <div className="mb-4 text-lg font-semibold text-[#233E7D]">
          Alerts Configurations
        </div>
        <nav>
          {menuItems.map((item) => (
            <MenuItem
              key={item.href}
              href={item.href}
              label={item.label}
              isActive={pathname.endsWith(item.href)}
            />
          ))}
        </nav>
      </div>
      <div className="w-full overflow-auto p-4">{children}</div>
    </div>
  );
};

export default Layout;