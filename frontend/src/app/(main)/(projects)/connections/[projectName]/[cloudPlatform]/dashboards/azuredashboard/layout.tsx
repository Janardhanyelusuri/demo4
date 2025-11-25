// src/app/(main)/(projects)/connections/[projectName]/[cloudPlatform]/dashboards/azuredashboard/layout.tsx

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


  
  useEffect(() => {
    if (item.label === "Tags" && isOpen) {
      const fetchTags = async () => {
        let retries = 0;
        const maxRetries = 3;
  
        setIsLoading(true);
        setError(null);
  
        while (retries < maxRetries) {
          try {
            const response = await axiosInstance.get(`${BACKEND}/tags/tags`);
            const data = response.data; // Accessing the data directly
            setTags(data);
            setIsLoading(false);
            return; // Exit the function on successful fetch
          } catch (error) {
            retries++;
            if (retries === maxRetries) {
              console.error("Error fetching tags:", error);
              setError("Failed to load tags. Please try again.");
              setIsLoading(false);
            }
          }
        }
      };
  
      fetchTags();
    }
  }, [item.label, isOpen]);
  
  
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
            const cloud_provider = pathParts[3];
            const project_id = pathParts[2];
  
            const body = {
              cloud_provider,
              query_type: "azure_dynamic_services_cost",
              project_id: project_id || "",
            };
  
            const response = await axiosInstance.post(`${BACKEND}/queries/queries`, body, {
              headers: {
                "Content-Type": "application/json",
                accept: "application/json",
              },
            });
  
            const result = response.data; // Accessing the data directly
            setDynamicServices(result.data);
            setIsLoading(false);
            return; // Exit the function on successful fetch
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

  if (item.label === "Tags") {
    return (
      <div>
        <div
          className={itemStyle}
          style={{ paddingLeft }}
          onClick={toggleOpen}
        >
          <motion.div
            className="flex-shrink-0 w-6 mr-2 text-cp-blue"
            animate={{ rotate: isOpen ? 90 : 0 }}
            transition={{ duration: 0.2 }}
          >
            <ChevronRight size={16} />
          </motion.div>
          <div className="flex-shrink-0 w-6 mr-2">
            {item.icon || <TagIcon size={16} className="text-blue-500" />}
          </div>
          <span className="text-sm truncate">{item.label}</span>
        </div>
        <AnimatePresence>
          {isOpen && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: "auto" }}
              exit={{ opacity: 0, height: 0 }}
              transition={{ duration: 0.3 }}
            >
              {isLoading ? (
                <div className="h-full w-full flex items-center justify-center">
                  <DotLoader />
                </div>
              ) : error ? (
                <div className="p-4 text-center text-red-500">{error}</div>
              ) : tags.length === 0 ? (
                <div className="p-4 text-center">No tags found</div>
              ) : (
                tags.map((tag, index) => (
                  <Link
                    key={index}
                    href={{
                      pathname: `${basePath}/azuredashboard/tags/${tag.tag_id}`,
                      query: { key: tag.key, value: tag.value }
                    }}
                    passHref
                  >
                    <div
                      className={`${itemStyle} ${
                        pathname === `${basePath}/azuredashboard/tags/${tag.tag_id}`
                          ? "bg-[#E0E5EF] text-[#0080FF] border-l-4 border-[#2F27CE]"
                          : ""
                      }`}
                      style={{
                        paddingLeft: `calc(${paddingLeft} + 32px)`,
                      }}
                    >
                      <span className="text-sm truncate">
                        {tag.key}: {tag.value}
                      </span>
                    </div>
                  </Link>
                ))
              )}
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    );
  }

  if (item.label === "Top Services") {
    return (
      <div>
        <div
          className={itemStyle}
          style={{ paddingLeft }}
          onClick={toggleOpen}
        >
          <motion.div
            className="flex-shrink-0 w-6 mr-2"
            animate={{ rotate: isOpen ? 90 : 0 }}
            transition={{ duration: 0.2 }}
          >
            <ChevronRight size={16} />
          </motion.div>
          <div className="flex-shrink-0 w-6 mr-2">
            {item.icon || <Server size={16} className="text-blue-500" />}
          </div>
          <span className="text-sm truncate">{item.label}</span>
        </div>
        <AnimatePresence>
          {isOpen && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: "auto" }}
              exit={{ opacity: 0, height: 0 }}
              transition={{ duration: 0.3 }}
            >
              {isLoading ? (
                <div className="h-full w-full flex items-center justify-center">
                  <DotLoader />
                </div>
              ) : error ? (
                <div className="p-4 text-center text-red-500">{error}</div>
              ) : dynamicServices.length === 0 ? (
                <div className="p-4 text-center">No services found</div>
              ) : (
                dynamicServices.map((service, index) => (
                  <Link
                    key={index}
                    href={{
                      pathname: `${basePath}/azuredashboard/services/${service.service_name}`,
                      query: { name: service.service_name, cost: service.cost }
                    }}
                    passHref
                  >
                    <div
                      className={`${itemStyle} ${
                        pathname === `${basePath}/azuredashboard/services/${service.service_name}`
                          ? "bg-[#E0E5EF] text-[#0080FF] border-l-4 border-[#2F27CE]"
                          : ""
                      }`}
                      style={{
                        paddingLeft: `calc(${paddingLeft} + 32px)`,
                      }}
                    >
                      <span className="text-cp-small truncate font-cp-normal">
                        {service.service_name}
                      </span>
                    </div>
                  </Link>
                ))
              )}
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    );
  }

  if (item.children) {
    return (
      <div>
        <div
          className={itemStyle}
          style={{ paddingLeft }}
          onClick={toggleOpen}
        >
          <motion.div
            className="flex-shrink-0 w-6 mr-2"
            animate={{ rotate: isOpen ? 90 : 0 }}
            transition={{ duration: 0.2 }}
          >
            <ChevronRight size={16} />
          </motion.div>
          <div className="flex-shrink-0 w-6 mr-2">
            {item.icon || <Folder size={16} className="text-blue-500" />}
          </div>
          <span className="text-sm truncate">{item.label}</span>
        </div>
        <AnimatePresence>
          {isOpen && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: "auto" }}
              exit={{ opacity: 0, height: 0 }}
              transition={{ duration: 0.3 }}
            >
              {item.children.map((child, index) => (
                <TreeItem
                  key={index}
                  item={child}
                  level={level + 1}
                  basePath={basePath}
                />
              ))}
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    );
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
      href: "/azuredashboard/AtAGlance",
    },
    {
      label: "Services Cost Table",
      icon: <Server size={16} className="mr-2 text-blue-500" />, 
      href: "/azuredashboard/AllServices",
    },
    {
      label: "Utilisation",
      icon: <Server size={16} className="mr-2 text-blue-500" />,
      href: "/azuredashboard/Utilisation",
    },
     {
         label: "Recommendations", // <-- FIX: Corrected spelling
         icon: <MoreHorizontal size={16} className="mr-2 text-blue-500" />,
         href: "/azuredashboard/recommendations", // <-- FIX: Corrected path
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
        <div className="mb-4 text-lg font-semibold">Azure Dashboard</div>
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