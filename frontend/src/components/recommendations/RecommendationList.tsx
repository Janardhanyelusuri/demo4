import React, { useState } from 'react';
import { NormalizedRecommendation } from "@/types/recommendations";
import RecommendationSummaryCard from './RecommendationSummaryCard';
import RecommendationCard from './RecommendationCard';
import { Button } from "@/components/ui/button";
import { ArrowLeft } from 'lucide-react';

interface RecommendationListProps {
    recommendations: NormalizedRecommendation[];
}

const RecommendationList: React.FC<RecommendationListProps> = ({ recommendations }) => {
    const [selectedIndex, setSelectedIndex] = useState<number | null>(null);
    const [filterSeverity, setFilterSeverity] = useState<'All' | 'High' | 'Medium' | 'Low'>('All');

    // Filter recommendations based on severity
    const filteredRecommendations = recommendations.filter(rec =>
        filterSeverity === 'All' || rec.severity === filterSeverity
    );

    // If a recommendation is selected, show the detail view
    if (selectedIndex !== null && filteredRecommendations[selectedIndex]) {
        const selectedRec = filteredRecommendations[selectedIndex];
        return (
            <div className="space-y-6">
                <Button
                    variant="ghost"
                    onClick={() => setSelectedIndex(null)}
                    className="flex items-center gap-2 text-gray-600 hover:text-gray-900 pl-0 hover:bg-transparent"
                >
                    <ArrowLeft className="w-4 h-4" />
                    Back to Recommendations
                </Button>

                <div className="animate-in fade-in slide-in-from-bottom-4 duration-300">
                    <RecommendationCard recommendation={selectedRec} />
                </div>
            </div>
        );
    }

    // Otherwise, show the list view
    return (
        <div className="space-y-6">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
                <p className="text-sm text-gray-600">
                    Found <span className="font-bold text-[#233E7D]">{filteredRecommendations.length}</span> resource
                    {filteredRecommendations.length !== 1 ? 's' : ''} with recommendations
                </p>

                {/* Severity Filter */}
                <div className="flex items-center space-x-2">
                    <span className="text-sm text-gray-500 font-medium">Filter by Severity:</span>
                    <div className="flex bg-gray-100 p-1 rounded-lg">
                        {(['All', 'High', 'Medium', 'Low'] as const).map((severity) => (
                            <button
                                key={severity}
                                onClick={() => setFilterSeverity(severity)}
                                className={`px-3 py-1 text-xs font-medium rounded-md transition-all ${filterSeverity === severity
                                    ? 'bg-white text-[#233E7D] shadow-sm'
                                    : 'text-gray-500 hover:text-gray-700'
                                    }`}
                            >
                                {severity}
                            </button>
                        ))}
                    </div>
                </div>
            </div>

            <div className="space-y-0">
                {filteredRecommendations.length === 0 ? (
                    <div className="text-center py-12 px-4 text-gray-500 bg-gray-50 rounded-md">
                        <p>No recommendations found with {filterSeverity} severity.</p>
                    </div>
                ) : (
                    filteredRecommendations.map((rec, index) => (
                        <RecommendationSummaryCard
                            key={`${rec.resourceId}-${index}`}
                            recommendation={rec}
                            onView={() => setSelectedIndex(index)}
                        />
                    ))
                )}
            </div>
        </div>
    );
};

export default RecommendationList;
