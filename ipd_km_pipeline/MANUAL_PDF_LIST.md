# Manual PDF Collection for K-M Curve Testing

## Strategy

Since automated PMC download has limited PDF availability, we'll use a hybrid approach:

1. **Known high-quality sources with K-M curves**:
   - NEJM oncology trials (free access)
   - Lancet Oncology open access
   - JAMA Oncology open access
   - Nature journals with open-access options
   - Specific PMC IDs known to have PDFs

2. **Criteria for selection**:
   - Clear K-M survival curves
   - Diverse cancer types (breast, lung, colorectal, hematologic, etc.)
   - Different journal styles/formatting
   - Varied time scales (months vs years)
   - Different numbers of curves (1-4 per panel)
   - Mix of:
     - Simple curves (2-curve comparison)
     - Complex multi-arm trials (3-4 curves)
     - Dotted reference lines
     - Censoring markers
     - Numbers-at-risk tables

---

## Alternative Download Approaches

### Approach 1: bioRxiv/medRxiv Preprints
- URL: https://www.medrxiv.org/
- All PDFs freely downloadable
- Search: "Kaplan-Meier" + "oncology" + "survival"
- High likelihood of K-M curves
- No paywall restrictions

### Approach 2: Specific Journal Open Access Collections
- **NEJM**: Some articles free after 6 months
- **JAMA Network Open**: Fully open access
- **BMC Cancer**: Open access
- **PLOS ONE**: Open access
- **Cancers (MDPI)**: Open access

### Approach 3: Known Clinical Trial Results
- Major trials with published K-M curves:
  - CheckMate trials (immunotherapy)
  - KEYNOTE trials (pembrolizumab)
  - CLEOPATRA (breast cancer)
  - CALGB trials (various cancers)

### Approach 4: Use the NEJM Paper We Already Have
- We already tested on NEJM diabetes paper (page 7)
- Find similar NEJM oncology papers
- Download directly from NEJM website

---

## Immediate Action Plan

1. **Search medRxiv/bioRxiv** (500+ oncology preprints with K-M curves)
2. **Manual curation**: Identify 40 diverse PDFs
3. **Download directly** from preprint servers
4. **Run batch processor** on curated collection

---

## Known Good PMC IDs (These Have PDFs)

Based on our earlier testing, these PMC IDs should have actual downloadable PDFs:

### From Earlier Searches:
- PMC9657332 (timeout but exists)
- (Need to identify more from successful downloads)

### Strategy to Find More:
- Search PMC for "open access" + "full text available" + specific keywords
- Filter by journals known to deposit PDFs (PLOS, BMC, MDPI)
- Use PMC OA subset: https://www.ncbi.nlm.nih.gov/pmc/tools/openftlist/

---

## Next Steps

1. Create medRxiv downloader script
2. Search for 50 oncology preprints with K-M curves
3. Download PDFs (should succeed - all preprints have PDFs)
4. Organize by cancer type and figure complexity
5. Run batch processor

This approach will give us:
- **Guaranteed PDF access** (preprints are always downloadable)
- **Diverse K-M curves** (different labs, styles, journals)
- **Real-world testing** (actual research papers, not synthetic)
