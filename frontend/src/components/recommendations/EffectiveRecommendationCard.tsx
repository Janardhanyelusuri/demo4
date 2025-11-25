import React from 'react';
import { Lightbulb } from 'lucide-react';
import { NormalizedRecommendation } from "@/types/recommendations";

interface EffectiveRecommendationCardProps {
    recommendation: NormalizedRecommendation;
}

const EffectiveRecommendationCard: React.FC<EffectiveRecommendationCardProps> = ({ recommendation: rec }) => {
    const effective = rec.effectiveRecommendation;

    return (
        <div className="h-full p-3 border-l-4 border-purple-500 rounded-lg shadow-md bg-white transition-shadow hover:shadow-lg flex flex-col">
            <div className="flex items-center space-x-2 mb-2">
                <Lightbulb className="w-4 h-4 text-purple-600" />
                <h3 className="text-sm font-semibold text-gray-800">Primary Recommendation</h3>
            </div>

            {/* Recommendation Text - Grows to fill space */}
            <div className="mb-3 flex-1">
                <p className="text-gray-700 text-xs leading-relaxed font-semibold">
                    {effective.text}
                </p>
                {effective.explanation && (
                    <p className="text-gray-600 text-xs leading-relaxed mt-2 pl-3 border-l-2 border-purple-300">
                        {effective.explanation}
                    </p>
                )}
            </div>

            {/* Saving Percentage - Stays at bottom */}
            <div className="bg-gradient-to-r from-purple-50 to-purple-100 rounded-lg p-2.5 flex items-center justify-between mt-auto">
                <span className="text-xs font-semibold text-gray-700">Estimated Saving:</span>
                <span className="text-xl font-bold text-purple-600">
                    {effective.saving_pct}%
                </span>
            </div>
        </div>
    );
};

export default EffectiveRecommendationCard;