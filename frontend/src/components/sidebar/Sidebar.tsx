import React from "react";
import {
  Tooltip,
  TooltipProvider,
  TooltipTrigger,
  TooltipContent,
} from "@/components/ui/tooltip";
import { usePathname, useRouter } from "next/navigation";
import Link from "next/link";
import Image from "next/image";
import LOGO from "@/assets/Cloud_Pulse_White.png";
import { menuOptions } from "@/lib/constant";
import clsx from "clsx";
import { LogOut, ChevronRight, ChevronLeft, Menu } from "lucide-react";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { signOut } from "next-auth/react";

type Props = {
  isOpen: boolean;
  setIsOpen: (isOpen: boolean) => void;
};

const Sidebar = ({ isOpen, setIsOpen }: Props) => {
  const pathName = usePathname();
  const router = useRouter();

  const isSelected = (href: string | string[]) => {
    if (Array.isArray(href)) {
      return href.some((link) => pathName.startsWith(link));
    }
    return pathName === href || pathName.startsWith(href);
  };

  const renderIcon = (
    Component: React.ComponentType<any>,
    selected: boolean
  ) => {
    return (
      <Component
        size={22}
        color={selected ? "#D82026" : "#fff"}
        className={clsx(
          "transition-colors",
          selected ? "text-[#D82026]" : "text-white"
        )}
      />
    );
  };

  const handleLogout = async () => {
    localStorage.clear();
    document.cookie.split(";").forEach((c) => {
      document.cookie = c
        .replace(/^ +/, "")
        .replace(/=.*/, "=;expires=" + new Date().toUTCString() + ";path=/");
    });
    await signOut({ redirect: false });
    router.push("/login");
  };

  const handleToggleSidebar = () => {
    setIsOpen(!isOpen);
  };

  // Updated menu options with new names
  const updatedMenuOptions = menuOptions.map((option) => {
    if (option.name === "Home") {
      return { ...option, name: "Dashboards" };
    } else if (option.name === "Notification") {
      return { ...option, name: "Connections" };
    }
    return option;
  });

  return (
    <>
      <nav
        className={`fixed left-0 flex flex-col gap-10 py-6 px-2 transition-all duration-300 ease-in-out ${
          isOpen ? "translate-x-0" : "-translate-x-full"
        }`}
        style={{
          top: 'calc(72px + (100vh - 72px)/2 - 336px)',
          height: '672px',
          backgroundColor: "#233E7D",
          boxShadow:
            "0 10px 30px rgba(0, 0, 0, 0.08), 0 6px 10px rgba(0, 0, 0, 0.05)",
          borderRight: "1px solid #E0E5EF",
          borderTopRightRadius: "16px",
          borderBottomRightRadius: "16px",
          width: "80px",
          zIndex: 40,
          position: "fixed",
        }}
      >
        <div className="absolute top-[20%] left-[15%] w-8 h-0.5 bg-gradient-to-r from-[#D82026] to-transparent opacity-15 rotate-12" />
        <div className="absolute top-[45%] right-[15%] w-6 h-0.5 bg-gradient-to-l from-[#D82026] to-transparent opacity-20 -rotate-12" />
        <div className="absolute top-[70%] left-[20%] w-5 h-0.5 bg-gradient-to-r from-[#D82026] to-transparent opacity-15 rotate-6" />
        <TooltipProvider>
          <div className="flex items-center flex-col gap-4 mt-4">
            {updatedMenuOptions.map((menuItem) => (
              <ul key={menuItem.name}>
                <Tooltip delayDuration={0}>
                  <TooltipTrigger asChild>
                    <li>
                      <Link
                        href={
                          Array.isArray(menuItem.href)
                            ? menuItem.href[0]
                            : menuItem.href
                        }
                        className={clsx(
                          "group h-10 w-10 flex items-center justify-center rounded-md cursor-pointer mt-2 text-sm font-medium transition-colors",
                          isSelected(menuItem.href)
                            ? "bg-white text-[#D82026] border-l-4 border-[#D82026]"
                            : "text-white hover:text-[#D82026] hover:bg-white/10"
                        )}
                        style={{
                          borderLeftWidth: isSelected(menuItem.href) ? 4 : 0,
                        }}
                        aria-label={menuItem.name}
                      >
                        {renderIcon(
                          menuItem.Component,
                          isSelected(menuItem.href)
                        )}
                      </Link>
                    </li>
                  </TooltipTrigger>
                  <TooltipContent
                    side="right"
                    className="bg-gray-800 text-white text-xs px-2 py-1 rounded"
                    sideOffset={5}
                  >
                    {menuItem.name}
                  </TooltipContent>
                </Tooltip>
              </ul>
            ))}
          </div>
          <div className="mt-auto px-4 py-2 flex flex-col gap-4 justify-center items-center">
            <Tooltip>
              <TooltipTrigger asChild>
                <button
                  onClick={handleLogout}
                  className="w-full flex justify-center items-center group relative"
                  aria-label="Logout"
                >
                  <LogOut color="#fff" />
                </button>
              </TooltipTrigger>
              <TooltipContent
                side="right"
                className="bg-gray-800 text-white text-xs px-2 py-1 rounded"
                sideOffset={5}
              >
                Logout
              </TooltipContent>
            </Tooltip>
            <div className="h-3" />
            <Avatar className="w-8 h-8">
              <AvatarImage src="https://st3.depositphotos.com/6672868/13701/v/450/depositphotos_137014128-stock-illustration-user-profile-icon.jpg" />
              <AvatarFallback className="bg-[#D82026] text-white">CP</AvatarFallback>
            </Avatar>
          </div>
        </TooltipProvider>
      </nav>
    </>
  );
};

export default Sidebar;