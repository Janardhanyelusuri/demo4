"use client";
import React from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";

type MenuItemProps = {
  href: string;
  label: string;
  isActive: boolean;
  basePath: string;
};

const MenuItem: React.FC<MenuItemProps> = ({
  href,
  label,
  isActive,
  basePath,
}) => {
  const fullPath = `${basePath}${href}`;
  return (
    <Link href={fullPath} passHref>
      <div
        className={`
          py-2 px-4 mb-2 rounded-sm transition-colors duration-200 text-xs font-medium
          ${
            isActive
              ? "bg-[#233E7D] text-white border-l-4 border-[#D82026]"
              : "hover:bg-[#233E7D]/10 text-[#233E7D] hover:text-[#D82026]"
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

  // Extract the base path
  const pathParts = pathname.split("/");
  const basePath = `/${pathParts.slice(1, 5).join("/")}`; // Adjust the slice range as needed

  const menuItems = [
    { href: "/alertRules", label: "Alert Rules" },
    { href: "/Points", label: "Points" },
  ];

  return (
    <div className="flex overflow-hidden h-screen">
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
              isActive={pathname === `${basePath}${item.href}`}
              basePath={basePath}
            />
          ))}
        </nav>
      </div>
      <div className="w-full overflow-auto p-4">{children}</div>
    </div>
  );
};

export default Layout;
