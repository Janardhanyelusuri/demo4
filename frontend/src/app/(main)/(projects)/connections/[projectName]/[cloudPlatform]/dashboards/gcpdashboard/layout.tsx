// src/app/(main)/(projects)/connections/[projectName]/[cloudPlatform]/dashboards/gcpdashboard/layout.tsx

"use client";

import React, { useState, useEffect } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  ChevronRight,
  Folder,
  LayoutDashboard,
  Server,
  Tag as TagIcon,
  MoreHorizontal,
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import DotLoader from "@/components/DotLoader";
import axiosInstance, { BACKEND } from "@/lib/api";

// ... (TagType, ServiceType, and TreeItemProps remain the same, assuming they are defined above)

type TagType = {
  tag_id: string;
  key: string;
  value: string;
};

type ServiceType = {
  service_name: string;
  cost: number;
};

type TreeItemProps = {
  item: {
    label: string;
    href?: string;
    icon?: React.ReactNode;
    children?: TreeItemProps["item"][];
  };
  level: number;
  basePath: string;
};

const TreeItem: React.FC<TreeItemProps> = ({ item, level, basePath }) => {
  const pathname = usePathname();
  const fullPath = item.href ? `${basePath}${item.href}` : "";
  const isActive = pathname === fullPath;

  const [isOpen, setIsOpen] = useState(false);
  const [tags, setTags] = useState<TagType[]>([]);
  const [dynamicServices, setDynamicServices] = useState<ServiceType[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // ... (useEffect for Tags remains the same, assuming it uses a GCP-compatible endpoint)
  useEffect(() => {
    if (item.label === "Tags" && isOpen) {
      // ... (Tag fetching logic remains the same)
    }
  }, [item.label, isOpen]);
  
  // Assuming this dynamic service fetching logic exists for GCP as well
  useEffect(() => {
    if (item.label === "Top Services" && isOpen) {
      const fetchServices = async () => {
        let retries = 0;
        const maxRetries = 3;

        setIsLoading(true);
        setError(null);

        while (retries < maxRetries) {
          try {
            const pathParts = pathname.split("/");
            const cloud_provider = pathParts[3]; // Should be 'gcp'
            const project_id = pathParts[2];

            const body = {
              cloud_provider,
              query_type: "gcp_dynamic_services_cost", // Assuming a corresponding GCP query type
              project_id: project_id || "",
            };

            const response = await axiosInstance.post(`${BACKEND}/queries/queries`, body, {
              headers: {
                "Content-Type": "application/json",
                accept: "application/json",
              },
            });

            const result = response.data; 
            setDynamicServices(result.data);
            setIsLoading(false);
            return; 
          } catch (error) {
            retries++;
            if (retries === maxRetries) {
              console.error("Error fetching services:", error);
              setError("Failed to load services. Please try again.");
              setIsLoading(false);
            }
          }
        }
      };

      fetchServices();
    }
  }, [item.label, isOpen, pathname]);


  useEffect(() => {
    if (typeof window !== "undefined") {
      const savedState = localStorage.getItem(item.label);
      if (savedState) {
        setIsOpen(JSON.parse(savedState));
      }
    }
  }, [item.label]);

  const toggleOpen = () => {
    setIsOpen(!isOpen);
    if (typeof window !== "undefined") {
      localStorage.setItem(item.label, JSON.stringify(!isOpen));
    }
  };

  const paddingLeft = `${level * 16}px`;
  const itemStyle = `flex items-center py-2 px-4 cursor-pointer transition-colors duration-200 rounded-md 
  ${isActive ? 'bg-cp-blue text-white border-l-4 border-cp-red' : 'text-cp-blue hover:text-cp-red hover:bg-cp-blue/10'}
`;
  
  // ... (Tags, Top Services, and Children rendering logic here, similar to Azure/AWS)
  if (item.label === "Tags") {
    // ... (logic for Tags)
  }

  if (item.label === "Top Services") {
    // ... (logic for Top Services)
  }

  if (item.children) {
    // ... (logic for folder items remains the same)
  } else {
    return (
      <Link href={fullPath} passHref>
        <div
          className={itemStyle}
          style={{
            paddingLeft: `calc(${paddingLeft} + ${isActive ? "14px" : "18px"})`,
          }}
        >
          <span className={`text-sm ${isActive ? "font-semibold" : ""}`}>
            {item.label}
          </span>
        </div>
      </Link>
    );
  }
};


type LayoutProps = {
  children: React.ReactNode;
};

const Layout: React.FC<LayoutProps> = ({ children }) => {
  const pathname = usePathname();
  const pathParts = pathname.split("/");
  const basePath = `/${pathParts.slice(1, 5).join("/")}`;
  const [sidebarWidth, setSidebarWidth] = useState(240);

  const treeStructure = [
    {
      label: "At A Glance",
      icon: <LayoutDashboard size={16} className="mr-2 text-blue-500" />,
      href: "/gcpdashboard/AtAGlance",
    },
    {
      label: "Services Cost Table",
      icon: <Server size={16} className="mr-2 text-blue-500" />,
      href: "/gcpdashboard/AllServices",
    },
    {
      label: "Utilisation",
      icon: <Server size={16} className="mr-2 text-blue-500" />,
      href: "/gcpdashboard/Utilisation",
    },
    {
      label: "Recommendations", // <-- FIX: Corrected spelling
      icon: <MoreHorizontal size={16} className="mr-2 text-blue-500" />,
      href: "/gcpdashboard/recommendations", // <-- FIX: Corrected path
    },
  ];

  const handleMouseDown = (e: React.MouseEvent<HTMLDivElement, MouseEvent>) => {
    e.preventDefault();
    const startX = e.clientX;
    const startWidth = sidebarWidth;

    const handleMouseMove = (e: MouseEvent) => {
      const newWidth = startWidth + (e.clientX - startX);
      if (newWidth > 150 && newWidth < 600) {
        setSidebarWidth(newWidth);
      }
    };

    const handleMouseUp = () => {
      document.removeEventListener("mousemove", handleMouseMove);
      document.removeEventListener("mouseup", handleMouseUp);
    };

    document.addEventListener("mousemove", handleMouseMove);
    document.addEventListener("mouseup", handleMouseUp);
  };

  return (
    <div className="flex overflow-hidden h-screen">
      <div
        className="h-full p-4 overflow-y-auto bg-white border-r border-t rounded-sm border-gray-200"
        style={{ width: sidebarWidth }}
      >
        <div className="mb-4 text-lg font-semibold">GCP Dashboard</div>
        <nav>
          {treeStructure.map((item, index) => (
            <TreeItem key={index} item={item} level={0} basePath={basePath} />
          ))}
        </nav>
      </div>
      <div
        className="resizer w-1 bg-gray-300 cursor-col-resize"
        onMouseDown={handleMouseDown}
      />
      <main className="flex-1 p-4 overflow-y-auto">{children}</main>
    </div>
  );
};

export default Layout;