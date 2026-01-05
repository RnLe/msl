import { MutableRefObject, useEffect, useRef, useState } from 'react'

type Size = { width: number; height: number }

export function useElementSize<T extends HTMLElement>(): [MutableRefObject<T | null>, Size] {
  const ref = useRef<T | null>(null)
  const [size, setSize] = useState<Size>({ width: 0, height: 0 })

  useEffect(() => {
    if (typeof ResizeObserver === 'undefined') {
      return
    }

    const node = ref.current
    if (!node) return

    const observer = new ResizeObserver((entries) => {
      const entry = entries[0]
      if (!entry) return
      const { width, height } = entry.contentRect
      setSize({ width, height })
    })

    observer.observe(node)
    return () => observer.disconnect()
  }, [])

  return [ref, size]
}
