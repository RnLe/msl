import { useCallback, useMemo } from 'react'
import type { KeyboardEvent as ReactKeyboardEvent } from 'react'
import clsx from 'clsx'
import { useElementSize } from './use-element-size'
import type { MaterialLibraryEntry } from './library'
import { getMaterialAccent, MATERIAL_CARD_PADDING_X, MATERIAL_CARD_PADDING_Y, MATERIAL_STICKER_SIZE, computeLatticeForFrequency } from './materialLibraryShared'
import { formatFrequencyRangeHz, formatWavelengthRangeMicron, mapMaterialWindows, getMaterialSpectralProfile } from './materialSpectra'

type MaterialLibraryCardProps = {
  entry: MaterialLibraryEntry
  active: boolean
  onSelect: (entry: MaterialLibraryEntry) => void
  onAlign?: (entry: MaterialLibraryEntry, frequencyHz: number) => void
  axisMaxNormalized: number
}

export function MaterialLibraryCard({ entry, active, onSelect, onAlign, axisMaxNormalized }: MaterialLibraryCardProps) {
  const [cardRef, cardSize] = useElementSize<HTMLDivElement>()
  const measuredHeight = Number.isFinite(cardSize.height) ? cardSize.height : 0
  const stickerSide = Math.max(Math.round(measuredHeight), MATERIAL_STICKER_SIZE)
  const accent = getMaterialAccent(entry.id)

  const profile = useMemo(() => getMaterialSpectralProfile(entry.id), [entry.id])
  const spectralWindows = useMemo(
    () =>
      mapMaterialWindows(profile, (window) => ({
        frequencyLabel: formatFrequencyRangeHz(window.frequencyHz),
        wavelengthLabel: formatWavelengthRangeMicron(window.wavelengthMicron),
      })),
    [profile]
  )

  const windowSummary = useMemo(() => {
    if (!spectralWindows.length) return null
    const summary = spectralWindows
      .map((window) => {
        const freq = window.frequencyLabel ?? ''
        const lam = window.wavelengthLabel ? ` (${window.wavelengthLabel})` : ''
        const combined = `${freq}${lam}`.trim()
        return combined || null
      })
      .filter((entryStr): entryStr is string => Boolean(entryStr))
      .join(' • ')
    return summary || null
  }, [spectralWindows])

  const windowLabel = spectralWindows.length > 1 ? 'Transparent Windows' : 'Transparent Window'

  const maxFrequencyHz = useMemo(() => {
    if (!profile?.windows?.length) return null
    let maxValue = 0
    profile.windows.forEach((window) => {
      if (!window.frequencyHz?.length) return
      const upper = Math.max(window.frequencyHz[0], window.frequencyHz[1])
      if (upper > maxValue) maxValue = upper
    })
    return maxValue > 0 ? maxValue : null
  }, [profile])

  const alignmentTarget = useMemo(() => {
    if (!onAlign || !maxFrequencyHz) return null
    return computeLatticeForFrequency(axisMaxNormalized, maxFrequencyHz)
  }, [axisMaxNormalized, maxFrequencyHz, onAlign])

  const handleCardClick = useCallback(() => {
    onSelect(entry)
  }, [entry, onSelect])

  const handleCardKeyDown = useCallback(
    (event: ReactKeyboardEvent<HTMLDivElement>) => {
      if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault()
        onSelect(entry)
      }
    },
    [entry, onSelect]
  )

  return (
    <div
      ref={cardRef}
      role="button"
      tabIndex={0}
      onClick={handleCardClick}
      onKeyDown={handleCardKeyDown}
      className={clsx(
        'w-full overflow-hidden border text-left transition',
        active
          ? 'border-[#3a3a3a] bg-[#1b1b1b] text-white'
          : 'border-[#1f1f1f] bg-[#111111] text-white/80 hover:border-[#2c2c2c] hover:bg-[#181818] hover:text-white'
      )}
    >
      <div className="flex h-full items-stretch">
        <div
          className="flex shrink-0 items-center justify-center text-lg font-semibold"
          style={{
            width: `${stickerSide}px`,
            height: `${stickerSide}px`,
            minHeight: MATERIAL_STICKER_SIZE,
            minWidth: MATERIAL_STICKER_SIZE,
            backgroundColor: accent.background,
            color: accent.text,
            boxShadow: '0 6px 16px rgba(0,0,0,0.35)',
            fontSize: '1.35rem',
            letterSpacing: '0.04em',
          }}
        >
          {entry.label}
        </div>
        <div
          className="flex-1 space-y-2"
          style={{
            padding: `${MATERIAL_CARD_PADDING_Y}px ${MATERIAL_CARD_PADDING_X}px`,
            minHeight: MATERIAL_STICKER_SIZE,
          }}
        >
          <div className="flex flex-wrap items-baseline justify-between gap-2">
            <div className="text-lg font-semibold" style={{ color: '#ffffff' }}>
              {entry.fullName}
            </div>
            <div className="text-lg font-mono" style={{ color: '#f3f3f3' }}>
              ε ≈ {entry.epsilon.toFixed(2)}
              <span className="inline-block" aria-hidden="true" style={{ width: '48px' }} />
              n ≈ {entry.refractiveIndex.toFixed(2)}
            </div>
          </div>
          <div className="text-sm" style={{ color: '#c2c2c2' }}>
            {entry.summary}
          </div>
          {entry.aliases?.length ? (
            <div className="text-[11px] uppercase tracking-[0.3em]" style={{ color: '#8f8f8f' }}>
              {entry.aliases.join(' / ')}
            </div>
          ) : null}
          {windowSummary ? (
            <div className="flex flex-wrap items-center gap-2 text-xs" style={{ color: '#a6a6a6' }}>
              <div className="flex flex-wrap items-center gap-2">
                <span className="font-semibold uppercase tracking-[0.2em]" style={{ color: '#8f8f8f' }}>
                  {windowLabel}
                </span>
                <span className="font-mono text-sm" style={{ color: '#f5f5f5' }}>
                  {windowSummary}
                </span>
              </div>
              {alignmentTarget ? (
                <button
                  type="button"
                  onClick={(event) => {
                    event.stopPropagation()
                    onAlign?.(entry, maxFrequencyHz as number)
                  }}
                  className="ml-auto rounded-full px-3.5 py-1.5 text-[12px] font-semibold text-white transition cursor-pointer hover:bg-white/25"
                  style={{ backgroundColor: 'rgba(255,255,255,0.14)' }}
                  aria-label={alignmentTarget ? `Set lattice constant to ${alignmentTarget.value} ${alignmentTarget.prefix}` : undefined}
                >
                  Set a = {alignmentTarget.value} {alignmentTarget.prefix}
                </button>
              ) : null}
            </div>
          ) : entry.designWindow ? (
            <div className="text-xs" style={{ color: '#a6a6a6' }}>
              <span className="font-semibold" style={{ letterSpacing: '0.08em' }}>
                λ window
              </span>
              <span className="ml-2 font-mono text-sm" style={{ color: '#f5f5f5' }}>
                {entry.designWindow}
              </span>
            </div>
          ) : null}
        </div>
      </div>
    </div>
  )
}
