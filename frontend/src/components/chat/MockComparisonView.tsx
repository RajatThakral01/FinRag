import React from 'react';
import { WarningBanner } from './WarningBanner';
import { HonestFailureCard } from './HonestFailureCard';
import { MessageBubble, type ChatMessage } from './MessageBubble';

export const MockComparisonView: React.FC = () => {
  const verifiedMessage: ChatMessage = {
    id: 'mock-verified',
    sender: 'assistant',
    content: 'Apple Inc. reported total revenue of $391,035 million in fiscal year 2024.',
    resolvedQuestion: 'What was Apple\'s total revenue in 2024?',
    questionWasResolved: true,
    cacheHit: true,
    sources: [
      {
        company: 'Apple Inc.',
        ticker: 'AAPL',
        year: '2024',
        section_name: 'Item 8. Financial Statements and Supplementary Data',
        table_name: 'Note 2 - Revenue',
        chunk_id: 'aapl_2024_item8_table_038_000',
        chunk_type: 'table',
      },
    ],
  };

  return (
    <div style={{ padding: '24px', maxWidth: '900px', margin: '0 auto', display: 'flex', flexDirection: 'column', gap: '24px' }}>
      <div style={{ borderBottom: '1px solid #2a2b2f', paddingBottom: '12px' }}>
        <h2 style={{ fontFamily: 'var(--font-display)', fontSize: '1.25rem', color: '#ededee' }}>
          Semantic Card States Comparison (Monochrome + Accent System)
        </h2>
        <p style={{ fontSize: '0.78rem', fontFamily: 'var(--font-mono)', color: '#8b8d94', marginTop: '4px' }}>
          Structure-driven distinction without color-coding: Solid Accent Left Border vs Dashed Metadata Gray Borders & Outline Icons.
        </p>
      </div>

      {/* 1. Verified Answer Card */}
      <div>
        <div style={{ fontSize: '0.7rem', fontFamily: 'var(--font-mono)', color: '#8ca0c7', textTransform: 'uppercase', marginBottom: '8px', letterSpacing: '0.04em' }}>
          1. Verified Answer Card (Thin Solid Accent Border #8CA0C7)
        </div>
        <MessageBubble message={verifiedMessage} />
      </div>

      {/* 2. Low-Confidence Warning Card */}
      <div>
        <div style={{ fontSize: '0.7rem', fontFamily: 'var(--font-mono)', color: '#8b8d94', textTransform: 'uppercase', marginBottom: '8px', letterSpacing: '0.04em' }}>
          2. Low-Confidence Warning Card (Dashed Gray Border #8B8D94 + AlertTriangle + "Low confidence")
        </div>
        <div style={{ background: '#1a1c20', padding: '16px', borderRadius: '6px', border: '1px solid #2a2b2f' }}>
          <WarningBanner errorMessage="Answer generated from best available context. Retrieval confidence was low — verify with source document." />
          <p style={{ fontSize: '0.88rem', color: '#ededee', lineHeight: '1.5' }}>
            According to available excerpts, total net sales for fiscal year 2024 were reported as $391,035 million.
          </p>
        </div>
      </div>

      {/* 3. Honest-Failure Card */}
      <div>
        <div style={{ fontSize: '0.7rem', fontFamily: 'var(--font-mono)', color: '#8b8d94', textTransform: 'uppercase', marginBottom: '8px', letterSpacing: '0.04em' }}>
          3. Honest-Failure Card (Dashed Gray Border #8B8D94 + AlertCircle + "Unverified")
        </div>
        <div style={{ background: '#1a1c20', padding: '16px', borderRadius: '6px', border: '1px solid #2a2b2f' }}>
          <HonestFailureCard
            answerSource="hallucination_exhausted"
            finalAnswer="I am unable to verify this financial figure against the provided SEC 10-K document excerpts with sufficient numerical confidence."
          />
        </div>
      </div>
    </div>
  );
};
