"""Genre-based music recommendation engine.

Given a predicted genre, this module suggests similar genres along with
example artists and playlists so users can explore related music.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Genre similarity graph
# Keys are genre labels; values are ordered lists of related genres
# (most similar first).
# ---------------------------------------------------------------------------

_GENRE_SIMILARITY: dict[str, list[str]] = {
    "blues":     ["jazz", "rock", "country", "reggae", "classical"],
    "classical": ["jazz", "blues", "country", "pop", "reggae"],
    "country":   ["blues", "rock", "pop", "jazz", "classical"],
    "disco":     ["pop", "reggae", "hip-hop", "rock", "jazz"],
    "hip-hop":   ["reggae", "disco", "pop", "blues", "jazz"],
    "jazz":      ["blues", "classical", "reggae", "country", "pop"],
    "metal":     ["rock", "blues", "classical", "pop", "country"],
    "pop":       ["disco", "rock", "country", "hip-hop", "reggae"],
    "reggae":    ["hip-hop", "blues", "jazz", "disco", "pop"],
    "rock":      ["metal", "blues", "country", "pop", "disco"],
}

# Short explanations for common genre pairings
_SIMILARITY_REASONS: dict[tuple[str, str], str] = {
    ("blues", "jazz"):       "Both share improvisation and expressive phrasing",
    ("blues", "rock"):       "Rock evolved directly from the blues tradition",
    ("blues", "country"):    "Deep Appalachian and Delta roots in common",
    ("classical", "jazz"):   "Complex harmony and instrumental virtuosity",
    ("country", "rock"):     "Shared guitar-driven sound and storytelling",
    ("disco", "pop"):        "Both prioritise danceable hooks and polished production",
    ("hip-hop", "reggae"):   "Strong rhythmic foundation and socially conscious lyrics",
    ("jazz", "blues"):       "Blues is the direct ancestor of jazz",
    ("metal", "rock"):       "Metal is a high-energy descendant of rock",
    ("pop", "disco"):        "Disco laid the groundwork for modern pop production",
    ("reggae", "hip-hop"):   "Hip-hop sampled heavily from reggae and dub",
    ("rock", "metal"):       "Metal amplifies rock's energy and aggression",
}

# Representative artists for each genre
_GENRE_ARTISTS: dict[str, list[str]] = {
    "blues":     ["B.B. King", "Muddy Waters", "Robert Johnson", "Etta James"],
    "classical": ["Mozart", "Beethoven", "Bach", "Chopin"],
    "country":   ["Johnny Cash", "Dolly Parton", "Willie Nelson", "Shania Twain"],
    "disco":     ["Donna Summer", "Bee Gees", "Gloria Gaynor", "Earth Wind & Fire"],
    "hip-hop":   ["Kendrick Lamar", "Jay-Z", "Eminem", "Nas"],
    "jazz":      ["Miles Davis", "John Coltrane", "Louis Armstrong", "Duke Ellington"],
    "metal":     ["Metallica", "Iron Maiden", "Black Sabbath", "Slayer"],
    "pop":       ["Michael Jackson", "Madonna", "Taylor Swift", "Ariana Grande"],
    "reggae":    ["Bob Marley", "Peter Tosh", "Jimmy Cliff", "Burning Spear"],
    "rock":      ["The Beatles", "Led Zeppelin", "The Rolling Stones", "Nirvana"],
}


class MusicRecommender:
    """Recommends genres and artists similar to a given genre.

    Usage
    -----
    >>> rec = MusicRecommender()
    >>> recommendations = rec.recommend("jazz", top_n=3)
    >>> for r in recommendations:
    ...     print(r["genre"], r["artists"])
    """

    def recommend(self, genre: str, top_n: int = 3) -> list[dict]:
        """Return a list of similar genre recommendations.

        Parameters
        ----------
        genre:
            The classified genre (must be one of the supported genres).
        top_n:
            Number of recommendations to return (default 3).

        Returns
        -------
        list[dict]
            Each element contains:
            - ``genre``   : genre label
            - ``artists`` : list of representative artists
            - ``reason``  : short explanation of similarity

        Raises
        ------
        ValueError
            If *genre* is not a recognised genre label.
        """
        genre = genre.lower()
        if genre not in _GENRE_SIMILARITY:
            supported = ", ".join(sorted(_GENRE_SIMILARITY))
            raise ValueError(
                f"Unknown genre '{genre}'. Supported genres: {supported}"
            )

        similar = _GENRE_SIMILARITY[genre][:top_n]
        results = []
        for sim_genre in similar:
            key = (genre, sim_genre)
            reason = _SIMILARITY_REASONS.get(
                key, f"Shares acoustic and cultural roots with {genre}"
            )
            results.append(
                {
                    "genre": sim_genre,
                    "artists": _GENRE_ARTISTS.get(sim_genre, []),
                    "reason": reason,
                }
            )
        return results
