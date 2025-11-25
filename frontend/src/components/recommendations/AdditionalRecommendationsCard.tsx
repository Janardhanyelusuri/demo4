// src/components/recommendations/AdditionalRecommendationsCard.tsx
import React from 'react';
import { CheckCircle2 } from 'lucide-react';
import { NormalizedRecommendation } from "@/types/recommendations";

interface AdditionalRecommendationsCardProps {
    recommendation: NormalizedRecommendation;
}

const AdditionalRecommendationsCard: React.FC<AdditionalRecommendationsCardProps> = ({ recommendation: rec }) => {
    const additionalRecs = rec.additionalRecommendations || [];

    return (
        <div className="h-full p-3 border-l-4 border-amber-500 rounded-lg shadow-md bg-white transition-shadow hover:shadow-lg flex flex-col">
            <div className="flex items-center space-x-2 mb-2">
                <CheckCircle2 className="w-4 h-4 text-amber-600" />
                <h3 className="text-sm font-semibold text-gray-800">Additional Recommendations</h3>
            </div>

            {/* Additional Recommendations List */}
            {additionalRecs.length > 0 ? (
                <div className="space-y-2 flex-1">
                    {additionalRecs.map((rec, index) => (
                        <div key={index} className="p-2.5 bg-amber-50 border border-amber-200 rounded-lg hover:bg-amber-100 transition-colors">
                            <div className="flex justify-between items-start gap-2">
                                <div className="flex-1">
                                    <p className="text-xs text-gray-700 leading-relaxed font-semibold">
                                        {rec.text}
                                    </p>
                                    {rec.explanation && (
                                        <p className="text-gray-600 text-xs leading-relaxed mt-1.5 pl-2 border-l-2 border-amber-300">
                                            {rec.explanation}
                                        </p>
                                    )}
                                </div>
                                <div className="flex-shrink-0">
                                    <span className="inline-block bg-amber-200 text-amber-900 font-semibold px-2 py-0.5 rounded-full text-xs whitespace-nowrap">
                                        {rec.saving_pct}%
                                    </span>
                                </div>
                            </div>
                        </div>
                    ))}
                </div>
            ) : (
                <p className="text-xs text-gray-500 italic text-center py-3">No additional recommendations available.</p>
            )}
        </div>
    );
};

export default AdditionalRecommendationsCard;