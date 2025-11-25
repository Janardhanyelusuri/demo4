import React from 'react';
import { NormalizedRecommendation } from "@/types/recommendations";
import { ArrowRight } from 'lucide-react';
import { Button } from "@/components/ui/button";

interface RecommendationSummaryCardProps {
    recommendation: NormalizedRecommendation;
    onView: () => void;
}

const getSeverityColor = (severity: 'High' | 'Medium' | 'Low') => {
    switch (severity) {
        case 'High': return 'text-red-600';
        case 'Medium': return 'text-orange-500';
        case 'Low': return 'text-green-600';
        default: return 'text-gray-500';
    }
};

// Helper function to get last N words from a string
const getLastWords = (text: string, wordCount: number = 3): string => {
    const words = text.trim().split(/\s+/);
    if (words.length <= wordCount) return text;
    return '...' + words.slice(-wordCount).join(' ');
};

const RecommendationSummaryCard: React.FC<RecommendationSummaryCardProps> = ({ recommendation: rec, onView }) => {
    const severityColor = getSeverityColor(rec.severity);
    const truncatedResourceId = getLastWords(rec.resourceId, 3);

    return (
        <div className="bg-white shadow-md rounded-md border p-4 mb-3 hover:shadow-lg transition-shadow">
            <div className="flex flex-col gap-3">
                {/* Header: Resource + Severity */}
                <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3 min-w-0 flex-1">
                        <span
                            className="text-sm font-semibold text-gray-900"
                            title={rec.resourceId}
                        >
                            {truncatedResourceId}
                        </span>
                        <span className={`text-xs font-medium ${severityColor} whitespace-nowrap`}>
                            {rec.severity}
                        </span>
                    </div>

                    {/* Savings */}
                    <div className="text-right hidden sm:block ml-4">
                        <p className="text-xs text-gray-400">Savings</p>
                        <p className="text-sm font-medium text-green-600">{rec.totalSavingPercent}%</p>
                    </div>
                </div>

                {/* Recommendation Text - Full Display */}
                <p className="text-sm text-gray-600">
                    {rec.effectiveRecommendation.text}
                </p>

                {/* Action Button */}
                <div className="flex justify-end">
                    <Button
                        onClick={onView}
                        variant="ghost"
                        size="sm"
                        className="text-[#233E7D] hover:text-[#1a2e5e] hover:bg-blue-50 gap-1"
                    >
                        View Details
                        <ArrowRight className="w-4 h-4" />
                    </Button>
                </div>
            </div>
        </div>
    );
};

export default RecommendationSummaryCard;
