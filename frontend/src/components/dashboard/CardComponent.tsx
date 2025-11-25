import React from "react";
import { Card, CardContent } from "@/components/ui/card";
import { DollarSign, TrendingUp } from "lucide-react";

interface CardComponentProps {
  label: string;
  value: number | string;
}

const CardComponent: React.FC<CardComponentProps> = ({ label, value }) => {
  const numericValue = typeof value === "string" ? parseFloat(value) : value;
  const isCloseToZero = Math.abs(numericValue) < 0.01; // Threshold check

  return (
    <Card className="w-full h-full overflow-hidden bg-cp-card border border-cp-border transition-all duration-300 hover:shadow-lg hover:scale-105 hover:border-cp-blue">
      <CardContent className="p-6">
        <div className="flex items-center justify-between mb-4">
          <div className="text-cp-small font-cp-semibold text-cp-text-gray uppercase tracking-wide">
            {label}
          </div>
          <div className="w-10 h-10 rounded-full bg-cp-blue/10 flex items-center justify-center">
            <DollarSign className="w-5 h-5 text-cp-blue" />
          </div>
        </div>
        <div className="text-3xl font-cp-bold text-cp-blue mb-2">
          {isCloseToZero
            ? "< $0.01"
            : `$${numericValue.toLocaleString(undefined, {
                minimumFractionDigits: 2,
                maximumFractionDigits: 2,
              })}`}
        </div>
      </CardContent>
      <div className="h-1 w-full bg-gradient-to-r from-cp-blue to-cp-success"></div>
    </Card>
  );
};

export default CardComponent;
