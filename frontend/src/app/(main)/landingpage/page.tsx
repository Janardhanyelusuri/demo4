"use client";

import React, { useEffect } from "react";
import Welcome from "@/assets/Cloud_Pulse.png";
import LandingPage from "@/assets/LandingGraphic.png"; // Updated to use PNG for better compatibility
import DropGraphics from "@/assets/DropGraphics.webp";
import Image from "next/image";
import { Button } from "@/components/ui/button";
import { useRouter } from "next/navigation";
import { useSession } from "next-auth/react";

const Page: React.FC = () => {
  const { data: session, status } = useSession();
  const router = useRouter();

  useEffect(() => {
    if (session?.accessToken) {
      localStorage.setItem("accessToken", session.accessToken);
    }
  }, [session]);

  const handleExistingProjectClick = () => {
    router.push("/connections");
  };

  const handleExistingDashboardsClick = () => {
    router.push("/dashboard-home");
  };

  return (
    <div className="fixed inset-0 flex justify-center items-center bg-[#F9FEFF] overflow-hidden">
      <div className="flex flex-col items-center justify-center bg-white rounded-md shadow-lg border border-[#E0E5EF] px-12 py-10 mr-8 relative z-10">
        <div className="-mt-8 -mb-8">
          <Image
            src={Welcome}
            alt="Welcome to Cloud Pulse"
            width={216}
            height={216}
            priority
          />
        </div>
        <h1 className="text-3xl font-bold text-[#233E7D] mt-6 mb-2">
          Welcome!
        </h1>
        <p className="text-base text-[#6B7280] mb-8">
          Manage Connections and Dashboards
        </p>
        <div className="mt-8 flex gap-5">
          <Button
            onClick={handleExistingProjectClick}
            className="bg-[#233E7D] hover:bg-[#1a2d5c] text-white text-sm font-medium rounded-md h-10 px-4 py-2"
          >
            See Connections
          </Button>
          <Button
            onClick={handleExistingDashboardsClick}
            className="bg-[#233E7D] hover:bg-[#1a2d5c] text-white text-sm font-medium rounded-md h-10 px-4 py-2"
          >
            See Dashboards
          </Button>
        </div>
      </div>
      <div className="ml-8 relative z-10 flex items-center">
        <Image
          src={LandingPage}
          alt="Landing Page"
          width={576}
          height={576}
          priority
          className="max-w-none"
        />
      </div>
    </div>
  );
};

export default Page;