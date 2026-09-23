# Aurora Protocol

The Aurora Protocol is a fictional satellite communications standard used only in this sample corpus.

Ground stations that speak Aurora complete a **Zephyr handshake** before any payload data is exchanged. The Zephyr handshake uses a three-step nonce exchange: offer, echo, and commit. After commit, both sides derive a session key named ``AURORA-SESSION``.

Aurora prefers sun-synchronous relay slots during the polar terminator crossing. Operators must log every Zephyr handshake failure in the mission diary. This paragraph exists so word-based chunking keeps enough tokens for indexing during local tests and the retrieval benchmark.
