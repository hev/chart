from pathlib import Path


INDEX = Path(__file__).resolve().parent.parent / "web" / "static" / "index.html"


def test_static_ui_displays_source_license_and_safety_notice() -> None:
    source = INDEX.read_text()

    assert "Dataset and safety notice" in source
    assert "https://huggingface.co/datasets/zhengyun21/PMC-Patients" in source
    assert "CC-BY-NC-SA-4.0" in source
    assert "not raw EHR and not for clinical use" in source


def test_static_ui_is_wired_to_live_api_routes() -> None:
    source = INDEX.read_text()

    assert "apiJson('/api/search?' + search + '&top_k=' + FETCH_DEPTH + (agentic ? '&agentic=1' : ''), agentic ? AGENT_TIMEOUT_MS : undefined)" in source
    assert "'q=' + encodeURIComponent(q)" in source
    assert "apiJson('/api/similar/'" in source
    assert "apiJson('/api/facets')" in source
    assert "data.routing" in source
    assert "renderRouting(data" in source
    assert "renderEcho" not in source


def test_static_ui_renders_gateway_routing_decision_instead_of_inferring_route() -> None:
    source = INDEX.read_text()

    # The route is read from the gateway's echo, never inferred from leg presence.
    assert "const route = routing.route || routing.strategy;" in source
    assert "(data.hybrid ? 'fused' : 'hybrid_text')" not in source
    # A query with no routing is impossible (the backend enforces require_routing);
    # the only no-route response is an honest filter-only browse, badged as such.
    assert "badge.textContent = 'filtered browse';" in source


def test_static_ui_treats_non_ok_api_responses_as_failures() -> None:
    source = INDEX.read_text()

    assert "async function apiJson(url, timeoutMs = 30_000)" in source
    assert "if (!response.ok)" in source
    assert "data.detail || data.message || `HTTP ${response.status}`" in source
    assert "Search failed: ${escapeHtml(err.message || err)}" in source
    assert "Similar-patient lookup failed: ${escapeHtml(err.message || err)}" in source


def test_static_ui_uses_gateway_row_ids_for_similar_lookup() -> None:
    source = INDEX.read_text()

    assert "const rowId = row.id || row['$id'] || ''" in source
    assert "runSimilar(rowId)" in source
    assert "if (rowId)" in source


def test_static_ui_surfaces_gemma_event_labels_in_results() -> None:
    source = INDEX.read_text()

    assert "Array.isArray(row.events)" in source
    assert "row.discontinuation_reason" in source
    # The cascade labels are clickable filter tags now, not plain meta text.
    assert 'data-field="discontinuation_reason"' in source
    assert 'data-field="events"' in source


def test_static_ui_surfaces_facet_snapshot_row_count_once_in_the_rail_footer() -> None:
    source = INDEX.read_text()

    assert "provenance || {}" in source
    assert "prov.row_count" in source
    # The corpus size is shown once, in the rail footer — not repeated per facet.
    assert "cases indexed" in source
    assert "railfoot" in source


def test_static_ui_facets_are_clickable_filters_that_narrow_the_search() -> None:
    source = INDEX.read_text()

    # Each facet value is a real button that toggles an active filter…
    assert "'fval'" in source
    assert "function toggleFilter(" in source
    assert "const activeFilters = new Map();" in source
    # …and the filter rides along on the same search as repeated `f=field:value`.
    assert "'f=' + encodeURIComponent(field + ':' + value)" in source
    # Active filters are shown and individually removable.
    assert "function renderActive(" in source
    assert "Clear all" in source


def test_static_ui_renders_facet_prevalence_bars() -> None:
    source = INDEX.read_text()

    # Each facet value carries a prevalence bar, sized as its share of the facet's
    # most common value, so a cohort's distribution reads at a glance.
    assert ".fbar {" in source
    assert 'class="fbar"' in source
    assert 'style="width:${pct.toFixed(1)}%"' in source
    assert "rows.reduce((m, v) => Math.max(m, Number(v.count) || 0), 0)" in source


def test_static_ui_hides_empty_facet_sections() -> None:
    source = INDEX.read_text()

    # UDF-writeback facets (events / specialty / diagnosis) render nothing until the
    # enrichment cascade populates them, instead of showing a bare empty heading.
    assert "const hasContent = (f) =>" in source
    # Coverage booleans (has_med_discontinuation) inform the rail foot but are
    # excluded from rendered facet sections.
    assert "filter((f) => !HIDDEN_FIELDS.has(f)).filter(hasContent)" in source


def test_static_ui_surfaces_declared_chip_expected_route() -> None:
    source = INDEX.read_text()

    assert "ex.expected_route" in source
    assert "Expected: ${ex.expected_route}." in source


def test_static_ui_fetches_live_facet_counts_via_scan_after_results() -> None:
    source = INDEX.read_text()

    # Counts are a SEPARATE call fired after the ranked results render — never
    # blocking them (the scan is slower than top-k).
    assert "function refreshCounts(" in source
    assert "refreshCounts(search, routing.route || routing.strategy || '')" in source
    # The shelf-shaped spelling: /api/facets?q= re-scopes the rail to this search.
    assert "'/api/facets?' + search" in source
    # A stale scan landing after a newer search must be discarded.
    assert "let countsSeq = 0;" in source
    assert "if (seq !== countsSeq) return;" in source
    # The live match-count headlines the rail.
    assert "let liveCounts = null;" in source
    assert "matching ${total === 1 ? 'case' : 'cases'}" in source


def test_static_ui_offers_an_agentic_search_toggle_and_renders_the_agent_plan() -> None:
    source = INDEX.read_text()

    # The toggle routes the query through the reasoning loop instead of Auto.
    assert '<input type="checkbox" id="agentic"' in source
    assert "&agentic=1" in source
    # The inspector renders the agentic badge, the loop's spend, and the plan.
    assert "function renderAgentic(" in source
    assert "badge.className = 'badge agentic';" in source
    assert "agent.queries" in source
    assert "agent.deadlineHit" in source
    # Per-row provenance: which planned variant surfaced the row + graded relevance.
    assert "row['$agent']" in source
    assert "relevanceScore" in source
    # Facet filters don't apply to agentic search; the rail drops live counts.
    assert "don’t apply to agentic search" in source


def test_static_ui_routes_semantic_counts_to_a_vector_radius_and_marks_them_approximate() -> None:
    source = INDEX.read_text()

    # The gateway's route decision is passed to the counts call so a semantic query
    # counts a vector radius, not keywords; the estimate is flagged approximate (~).
    assert "'&route=' + encodeURIComponent(route)" in source
    assert "liveCounts.approximate" in source
    assert "${approx ? '~' : ''}" in source


def test_static_ui_paginates_results_client_side() -> None:
    source = INDEX.read_text()

    # Auto routing can't cursor-paginate (fused scores have no monotone bands), so a
    # deeper page is fetched once and walked client-side.
    assert "const PAGE_SIZE = 10;" in source
    assert "const FETCH_DEPTH = 50;" in source
    assert "&top_k=' + FETCH_DEPTH" in source
    assert "function renderPage(" in source
    assert "function renderPager(" in source
    assert "allRows.slice(start, start + PAGE_SIZE)" in source


def test_static_ui_links_back_to_hevlayer() -> None:
    source = INDEX.read_text()

    assert 'href="https://hevlayer.com"' in source
    assert source.count("https://hevlayer.com") >= 2


def test_static_ui_note_viewer_shows_full_note_and_every_cascade_label() -> None:
    source = INDEX.read_text()

    # The viewer is a native <dialog>, opened from each hit (button + snippet).
    assert 'id="noteview"' in source
    assert "function openNote(row)" in source
    assert "noteview.showModal()" in source
    assert 'noteBtn.textContent = ' in source and "openNote(row)" in source
    # Every cascade output renders — not just the events already on the card.
    assert "Gemma cascade classifications" in source
    assert "row.body_system" in source
    assert "row.chief_complaint" in source
    assert "row.has_adverse_event" in source
    assert "row.has_med_discontinuation" in source
    # Honesty: an unclassified note says so instead of showing a blank grid.
    assert "Not yet classified" in source
    # The safety notice repeats inside the viewer (full note text is on screen).
    assert "not raw EHR and not for clinical use." in source


def test_static_ui_note_viewer_highlights_gateway_lexical_tokens_only() -> None:
    source = INDEX.read_text()

    # The highlight vocabulary is the gateway's hybrid echo — captured per search,
    # cleared for agentic runs (per-variant token sets have no single vocabulary).
    assert "lastEcho = { route: routing.route || routing.strategy || '', hybrid: data.hybrid || null };" in source
    assert "lastEcho = null; // per-variant token sets" in source
    assert "lastEcho.hybrid.tokens" in source
    # Literal, whole-token occurrences get <mark>ed; the UI never re-tokenizes.
    assert "function highlightNote(text, tokens)" in source
    assert "<mark>" in source
    # A fuzzy-leg match (from the row's $fused provenance) is labeled honestly —
    # the gateway doesn't echo the matched variant yet, so the UI doesn't guess.
    assert "startsWith('fuzzy:')" in source
    assert "mterm fuzzy" in source
    assert "row['$fused']" in source


def test_static_ui_corpus_rail_survives_a_failed_load_and_renders_without_a_query() -> None:
    source = INDEX.read_text()

    # The corpus rail (total facet counts, no search term) retries with capped
    # backoff instead of giving up for the session on one failed load…
    assert "window.setTimeout(loadFacets, facetRetryMs)" in source
    assert "facetRetryMs = Math.min(facetRetryMs * 2, 30_000);" in source
    assert "Corpus facets unavailable — retrying…" in source
    # …and the rail can render from live per-search counts while the corpus
    # snapshot set is missing, instead of blanking on `!facetData`.
    assert "const corpus = facetData || {};" in source
    assert "if (!facetData && !liveCounts)" in source


def test_static_ui_rail_state_is_declared_before_the_top_level_facet_load() -> None:
    """The module top-level `await loadFacets()` runs mid-file; any `let` it (or
    renderRail) touches must be declared ABOVE that call. A later declaration is
    a temporal-dead-zone ReferenceError that kills the whole module on load —
    no rail, ever (shipped once: chart-search-20260703d)."""
    source = INDEX.read_text()

    load_call = source.index("await loadFacets();")
    for state in ("let facetLoading", "let facetError", "let facetRetryMs", "let facetData", "let liveCounts"):
        assert source.index(state) < load_call, f"{state} declared after the top-level loadFacets() call"
