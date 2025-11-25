// src/components/recommendations/MetricsCard.tsx
import React from 'react';
import { TrendingUp } from 'lucide-react';
import { NormalizedRecommendation } from "@/types/recommendations";

interface MetricsCardProps {
    recommendation: NormalizedRecommendation;
}

const MetricsCard: React.FC<MetricsCardProps> = ({ recommendation: rec }) => {
    const baseMetrics = rec.baseOfRecommendations || [];
    const monthly = rec.costForecasting.monthly;
    const annually = rec.costForecasting.annually;

    return (
        <div className="p-3 border-l-4 border-blue-500 rounded-lg shadow-md bg-white transition-shadow hover:shadow-lg">
            <div className="flex items-center space-x-2 mb-2">
                <TrendingUp className="w-4 h-4 text-blue-600" />
                <h3 className="text-sm font-semibold text-gray-800">Metrics & Forecasting</h3>
            </div>

            {/* Metrics Basis Section */}
            <div className="mb-3">
                <h4 className="text-xs font-semibold text-gray-700 mb-2">Base of Recommendations:</h4>
                <div className="space-y-1.5">
                    {baseMetrics.length > 0 ? (
                        baseMetrics.map((metric, index) => (
                            <div key={index} className="flex justify-between items-center p-2 bg-gray-50 rounded">
                                <span className="text-xs text-gray-700">{metric}</span>
                                <span className="text-[10px] font-mono bg-blue-100 text-blue-800 px-1.5 py-0.5 rounded">
                                    {/* Extract unit if present (e.g., "(GiB)" from "UsedCapacity (GiB)_Avg") */}
                                    {metric.match(/\(.*?\)/)?.[0] || 'N/A'}
                                </span>
                            </div>
                        ))
                    ) : (
                        <p className="text-xs text-gray-500 italic">No metrics information available.</p>
                    )}
                </div>
            </div>

            {/* Cost Forecasting Section */}
            <div className="border-t pt-3">
                <h4 className="text-xs font-semibold text-gray-700 mb-2">Cost Forecast:</h4>
                <div className="grid grid-cols-2 gap-2">
                    <div className="p-2 bg-green-50 rounded">
                        <p className="text-[10px] text-gray-600 mb-0.5">Monthly Forecast</p>
                        <p className="text-lg font-bold text-green-600">
                            ${monthly.toFixed(2)}
                        </p>
                    </div>
                    <div className="p-2 bg-green-50 rounded">
                        <p className="text-[10px] text-gray-600 mb-0.5">Annual Forecast</p>
                        <p className="text-lg font-bold text-green-600">
                            ${annually.toFixed(2)}
                        </p>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default MetricsCard;