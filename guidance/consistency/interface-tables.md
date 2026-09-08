# Root API-table review

Reviewed the tables against their defining Rust signatures after each author froze the chapter. The tables now distinguish successful values from fallible `Result` returns and distinguish returned models from updates performed through `&mut self`. Shape descriptions stay beside the actual input and output types.

Corrections cover Chapters 1–6, 9–15, 17–21, 23–24, 27, 29–42, 44–46, 49–50, and 53. The Chapter 31 table separately describes scalar CPU methods and buffer-based WGSL kernels: its serial backward kernel writes one mean batch loss plus the mean parameter gradient, and its update kernel reads already-computed gradients. Chapter 35 names the actual Attention/AttentionGradients result records. Chapter 45 separates the environment transition from value and policy updates.

All affected lesson tables use the existing horizontal table wrapper for narrow screens. This table pass changes documentation only; numerical behavior is checked in each chapter's integrated validation record. A Luna High reviewer independently checked the broad return-type scan and the CPU/WGSL distinction.
