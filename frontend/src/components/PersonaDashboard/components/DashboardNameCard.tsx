"use client";
import React from "react";
import { Card, CardContent } from "@/components/ui/card";

type DashboardNameCardProps = {
  title: string;
};

const DashboardNameCard: React.FC<DashboardNameCardProps> = ({ title }) => {
  return (
    <div>
      <div
        className="relative w-full border border-[#E0E5EF] bg-[#FFFFFF] text-[#233E7D] overflow-hidden transition-all duration-300 rounded-md shadow hover:shadow-lg hover:border-[#233E7D]"
      >
        <CardContent className="p-4">
          <div className="flex items-center justify-between mb-2">
            <div className="flex-grow text-left">
              <div className="text-2xl font-bold tracking-tight text-[#233E7D]">
                Dashboard: <span className="font-semibold text-[#233E7D]">{title}</span>
              </div>
            </div>
          </div>
        </CardContent>
        <div className="absolute bottom-0 left-0 w-full h-1 bg-gradient-to-r from-[#D82026] to-[#233E7D]" />
      </div>
    </div>
  );
};

export default DashboardNameCard;
