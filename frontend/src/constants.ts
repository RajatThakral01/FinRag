/**
 * frontend/src/constants.ts
 * Shared constants used across multiple components.
 * Single source of truth — edit here, not in individual components.
 */

export interface ExampleQuery {
  label: string;
  text: string;
}

/** Full set of example queries shown on the welcome screen. */
export const EXAMPLE_QUERIES: ExampleQuery[] = [
  { label: 'FACTUAL',     text: "What was Apple's total revenue in fiscal 2024?" },
  { label: 'FACTUAL',     text: "What was NVIDIA's R&D expense in 2024?" },
  { label: 'COMPARATIVE', text: 'Compare Apple and Microsoft operating income' },
  { label: 'COMPARATIVE', text: 'Which company had the highest R&D spend in 2024?' },
  { label: 'CALCULATION', text: "What was Apple's gross margin percentage?" },
  { label: 'CALCULATION', text: "What was Tesla's revenue growth from 2023 to 2024?" },
];

/** Subset shown as quick-access chips in the chat input bar. */
export const SAMPLE_QUERIES: string[] = [
  "What was Apple's total revenue in 2024?",
  "What was NVIDIA's R&D expense in 2024?",
  "Compare Apple and Microsoft operating income",
];
