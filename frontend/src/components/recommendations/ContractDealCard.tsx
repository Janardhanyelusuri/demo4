// src/components/recommendations/ContractDealCard.tsx
import React, { useState } from 'react';
import { TrendingUp, TrendingDown, Minus, ChevronDown, ChevronUp } from 'lucide-react';
import { NormalizedRecommendation } from "@/types/recommendations";

interface ContractDealCardProps {
    recommendation: NormalizedRecommendation;
    isExpanded?: boolean;
}

const ContractDealCard: React.FC<ContractDealCardProps> = ({ recommendation: rec, isExpanded = false }) => {
    const [expanded, setExpanded] = useState(isExpanded);
    const deal = rec.contractDeal;

    // Determine assessment styling
    const getAssessmentStyles = (assessment: string) => {
        switch (assessment.toLowerCase()) {
            case 'good':
                return {
                    bgLight: 'bg-green-50',
                    border: 'border-green-200',
                    badge: 'bg-green-100 text-green-800',
                    text: 'text-green-700',
                    headerBg: 'bg-green-100',
                    icon: <TrendingUp className="w-5 h-5 text-green-600" />
                };
            case 'bad':
                return {
                    bgLight: 'bg-red-50',
                    border: 'border-red-200',
                    badge: 'bg-red-100 text-red-800',
                    text: 'text-red-700',
                    headerBg: 'bg-red-100',
                    icon: <TrendingDown className="w-5 h-5 text-red-600" />
                };
            default:
                return {
                    bgLight: 'bg-gray-50',
                    border: 'border-gray-200',
                    badge: 'bg-gray-100 text-gray-800',
                    text: 'text-gray-700',
                    headerBg: 'bg-gray-100',
                    icon: <Minus className="w-5 h-5 text-gray-600" />
                };
        }
    };

    const styles = getAssessmentStyles(deal.assessment);

    return (
        <div className="bg-white rounded-lg border border-cyan-200 shadow-sm hover:shadow-md transition-shadow overflow-hidden">
            {/* Header Button - Reduced padding */}
            <button
                onClick={() => setExpanded(!expanded)}
                className="w-full p-3 flex items-center justify-between hover:bg-cyan-50 transition-colors"
            >
                <div className="flex items-center space-x-2">
                    <div className={`${styles.headerBg} p-1.5 rounded-lg`}>
                        {styles.icon}
                    </div>
                    <div className="text-left">
                        <h3 className="text-xs font-semibold text-gray-800">Contract Deal</h3>
                        <p className={`text-[10px] font-semibold ${styles.text}`}>
                            {deal.assessment.charAt(0).toUpperCase() + deal.assessment.slice(1)} · {deal['for sku']}
                        </p>
                    </div>
                </div>
                <div className="flex items-center space-x-1.5">
                    <span className={`text-[10px] font-semibold ${styles.badge} px-1.5 py-0.5 rounded`}>
                        {deal.annual_saving_pct > 0 ? '+' : ''}{deal.annual_saving_pct}%
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
                <div className="border-t border-cyan-200 p-3 space-y-2.5">
                    {/* Assessment Details */}
                    <div className={`p-2.5 rounded-lg ${styles.bgLight} border ${styles.border}`}>
                        <div className="flex justify-between items-start">
                            <div>
                                <p className="text-[10px] text-gray-600 font-semibold mb-0.5">Assessment</p>
                                <p className={`text-sm font-bold capitalize ${styles.text}`}>
                                    {deal.assessment}
                                </p>
                            </div>
                            <span className={`${styles.badge} px-2 py-0.5 rounded-full text-[10px] font-semibold`}>
                                {deal['for sku']}
                            </span>
                        </div>
                    </div>

                    {/* Reason */}
                    <div className="p-2.5 bg-gray-50 rounded-lg">
                        <p className="text-[10px] text-gray-600 font-semibold mb-1">Reason</p>
                        <p className="text-xs text-gray-700 leading-relaxed">{deal.reason}</p>
                    </div>

                    {/* Savings/Loss Metrics */}
                    <div>
                        <p className="text-[10px] text-gray-600 font-semibold mb-2">Monthly & Annual Impact</p>
                        <div className="grid grid-cols-2 gap-2">
                            <div className={`p-2.5 rounded-lg border ${deal.monthly_saving_pct > 0 ? 'bg-green-50 border-green-200' : 'bg-red-50 border-red-200'}`}>
                                <p className="text-[10px] text-gray-600 font-semibold mb-1">Monthly</p>
                                <p className={`text-lg font-bold ${deal.monthly_saving_pct > 0 ? 'text-green-600' : 'text-red-600'}`}>
                                    {deal.monthly_saving_pct > 0 ? '+' : ''}{deal.monthly_saving_pct}%
                                </p>
                            </div>
                            <div className={`p-2.5 rounded-lg border ${deal.annual_saving_pct > 0 ? 'bg-green-50 border-green-200' : 'bg-red-50 border-red-200'}`}>
                                <p className="text-[10px] text-gray-600 font-semibold mb-1">Annual</p>
                                <p className={`text-lg font-bold ${deal.annual_saving_pct > 0 ? 'text-green-600' : 'text-red-600'}`}>
                                    {deal.annual_saving_pct > 0 ? '+' : ''}{deal.annual_saving_pct}%
                                </p>
                            </div>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
};

export default ContractDealCard;