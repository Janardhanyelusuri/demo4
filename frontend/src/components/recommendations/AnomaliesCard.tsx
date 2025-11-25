// src/components/recommendations/AnomaliesCard.tsx
import React, { useState } from 'react';
import { AlertTriangle, ChevronDown, ChevronUp } from 'lucide-react';
import { NormalizedRecommendation } from "@/types/recommendations";

interface AnomaliesCardProps {
    recommendation: NormalizedRecommendation;
    isExpanded?: boolean;
}

const AnomaliesCard: React.FC<AnomaliesCardProps> = ({ recommendation: rec, isExpanded = false }) => {
    const [expanded, setExpanded] = useState(isExpanded);
    const anomalies = rec.anomalies || [];

    return (
        <div className="bg-white rounded-lg border border-red-200 shadow-sm hover:shadow-md transition-shadow overflow-hidden">
            {/* Header Button - Reduced padding */}
            <button
                onClick={() => setExpanded(!expanded)}
                className="w-full p-3 flex items-center justify-between hover:bg-red-50 transition-colors"
            >
                <div className="flex items-center space-x-2">
                    <div className="bg-red-100 p-1.5 rounded-lg">
                        <AlertTriangle className="w-4 h-4 text-red-600" />
                    </div>
                    <div className="text-left">
                        <h3 className="text-xs font-semibold text-gray-800">Anomalies</h3>
                        <p className="text-[10px] text-gray-500">{anomalies.length} found</p>
                    </div>
                </div>
                <div className="flex items-center space-x-1.5">
                    <span className="text-[10px] font-semibold text-red-600 bg-red-50 px-1.5 py-0.5 rounded">
                        {anomalies.length}
                    </span>
                    {expanded ? (
                        <ChevronUp className="w-4 h-4 text-gray-600" />
                    ) : (
                        <ChevronDown className="w-4 h-4 text-gray-600" />
                    )}
                </div>
            </button>

            {/* Content - Collapsible with reduced padding */}
            {expanded && (
                <div className="border-t border-red-200 p-3">
                    {anomalies.length > 0 ? (
                        <div className="space-y-2">
                            {anomalies.map((anomaly, index) => (
                                <div key={index} className="p-2.5 bg-red-50 border border-red-200 rounded-lg">
                                    <div className="grid grid-cols-2 gap-3 mb-2">
                                        <div>
                                            <p className="text-[10px] text-gray-600 font-semibold mb-0.5">Metric</p>
                                            <p className="text-xs font-mono text-gray-800">{anomaly.metric_name}</p>
                                        </div>
                                        <div>
                                            <p className="text-[10px] text-gray-600 font-semibold mb-0.5">Date</p>
                                            <p className="text-xs font-mono text-gray-800">
                                                {anomaly.timestamp.split('T')[0] || anomaly.timestamp}
                                            </p>
                                        </div>
                                    </div>

                                    <div className="mb-2">
                                        <p className="text-[10px] text-gray-600 font-semibold mb-0.5">Value</p>
                                        <p className="text-lg font-bold text-red-600">{anomaly.value}</p>
                                    </div>

                                    <div className="bg-red-100 rounded p-2">
                                        <p className="text-[10px] text-gray-600 font-semibold mb-0.5">Reason</p>
                                        <p className="text-xs text-gray-700">{anomaly.reason_short}</p>
                                    </div>
                                </div>
                            ))}
                        </div>
                    ) : (
                        <p className="text-xs text-gray-500 italic text-center py-4">No anomalies detected.</p>
                    )}
                </div>
            )}
        </div>
    );
};

export default AnomaliesCard;