// src/components/recommendations/KPICard.tsx
import React from 'react';
import { NormalizedRecommendation } from "@/types/recommendations";
import { DollarSign, TrendingDown, AlertCircle, CheckCircle } from 'lucide-react';

interface KPICardProps {
    recommendation: NormalizedRecommendation;
}

const KPICard: React.FC<KPICardProps> = ({ recommendation: rec }) => {
    const anomalyCount = rec.anomalies?.length || 0;
    const additionalRecsCount = rec.additionalRecommendations?.length || 0;
    const hasContractDeal = rec.contractDeal ? true : false;

    return (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
            {/* KPI: Total Monthly Forecast */}
            <div className="bg-white rounded-lg border border-gray-200 p-3 hover:shadow-md transition-shadow">
                <div className="flex items-center justify-between">
                    <div className="min-w-0 flex-1">
                        <p className="text-[10px] font-semibold text-gray-500 uppercase tracking-wide">Monthly Forecast</p>
                        <p className="text-xl font-bold text-blue-600 mt-1 truncate">
                            ${rec.costForecasting.monthly.toFixed(2)}
                        </p>
                    </div>
                    <div className="bg-blue-100 p-2 rounded-lg flex-shrink-0 ml-2">
                        <DollarSign className="w-5 h-5 text-blue-600" />
                    </div>
                </div>
            </div>

            {/* KPI: Annual Forecast */}
            <div className="bg-white rounded-lg border border-gray-200 p-3 hover:shadow-md transition-shadow">
                <div className="flex items-center justify-between">
                    <div className="min-w-0 flex-1">
                        <p className="text-[10px] font-semibold text-gray-500 uppercase tracking-wide">Annual Forecast</p>
                        <p className="text-xl font-bold text-indigo-600 mt-1 truncate">
                            ${rec.costForecasting.annually.toFixed(2)}
                        </p>
                    </div>
                    <div className="bg-indigo-100 p-2 rounded-lg flex-shrink-0 ml-2">
                        <TrendingDown className="w-5 h-5 text-indigo-600" />
                    </div>
                </div>
            </div>

            {/* KPI: Total Saving Percentage */}
            <div className="bg-white rounded-lg border border-gray-200 p-3 hover:shadow-md transition-shadow">
                <div className="flex items-center justify-between">
                    <div className="min-w-0 flex-1">
                        <p className="text-[10px] font-semibold text-gray-500 uppercase tracking-wide">Total Savings</p>
                        <p className="text-xl font-bold text-green-600 mt-1">
                            {rec.totalSavingPercent}%
                        </p>
                    </div>
                    <div className="bg-green-100 p-2 rounded-lg flex-shrink-0 ml-2">
                        <CheckCircle className="w-5 h-5 text-green-600" />
                    </div>
                </div>
            </div>

            {/* KPI: Anomalies Count */}
            <div className="bg-white rounded-lg border border-gray-200 p-3 hover:shadow-md transition-shadow">
                <div className="flex items-center justify-between">
                    <div className="min-w-0 flex-1">
                        <p className="text-[10px] font-semibold text-gray-500 uppercase tracking-wide">Anomalies</p>
                        <p className="text-xl font-bold text-red-600 mt-1">
                            {anomalyCount}
                        </p>
                    </div>
                    <div className="bg-red-100 p-2 rounded-lg flex-shrink-0 ml-2">
                        <AlertCircle className="w-5 h-5 text-red-600" />
                    </div>
                </div>
            </div>
        </div>
    );
};

export default KPICard;