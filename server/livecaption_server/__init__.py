"""Central meeting-records server for livecaption.

Multi-user accounts + a shared meeting library. Local Mac capture agents push finalized
meetings here over an API token; users log in on the web to browse / search / annotate /
export their own and shared meetings. No audio capture and no models run here — it is a
plain web + SQLite service that runs on any machine (the company Linux box), no GPU/MLX.
"""
