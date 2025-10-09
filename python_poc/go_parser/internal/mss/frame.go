package mss

import (
	"encoding/binary"
	"errors"
	"hash/crc32"
	"io"
	"math"
)

const (
	MagicMONO uint32 = 0x4F4E4F4D // 'MONO' little-endian
	VersionV1 uint16 = 0x0001
)

// Half represents an IEEE-754 binary16 value (float16).
type Half uint16

// ToFloat32 converts a binary16 to float32.
func (h Half) ToFloat32() float32 {
	b := uint16(h)
	sign := (b >> 15) & 0x1
	exp := (b >> 10) & 0x1F
	man := b & 0x3FF

	if exp == 0 {
		if man == 0 {
			if sign == 1 {
				return float32(math.Copysign(0, -1))
			}
			return 0
		}
		// subnormal: (-1)^sign x 2^(-14) x (mantissa / 2^10)
		f := math.Ldexp(float64(man), -24)
		if sign == 1 {
			f = -f
		}
		return float32(f)
	} else if exp == 0x1F {
		if man == 0 {
			if sign == 1 {
				return float32(math.Inf(-1))
			}
			return float32(math.Inf(1))
		}
		return float32(math.NaN())
	}

	// normalized
	val := (1.0 + float64(man)/1024.0) * math.Pow(2, float64(int(exp)-15))
	if sign == 1 {
		val = -val
	}
	return float32(val)
}

type TopLogit struct {
	TokenID uint32  `json:"token_id"`
	Prob    float32 `json:"prob"`
}

type AttnEntry struct {
	Layer       uint16  `json:"layer"`
	Head        uint16  `json:"head"`
	TopTokenIdx uint16  `json:"top_token_idx"`
	Weight      float32 `json:"weight"`
}

type ConceptEntry struct {
	ConceptID uint16  `json:"concept_id"`
	Score     float32 `json:"score"`
}

type FrameV1 struct {
	Magic        uint32         `json:"magic"`
	Version      uint16         `json:"version"`
	HeaderLen    uint16         `json:"header_len"`
	PromptNonce  uint64         `json:"prompt_nonce"`
	TokenIndex   uint32         `json:"token_index"`
	ChosenID     uint32         `json:"chosen_id"`
	TSC          uint64         `json:"tsc"`
	TopK         uint16         `json:"topk"`
	TopLogits    []TopLogit     `json:"top_logits"`
	AttnCount    uint16         `json:"attn_count"`
	Attn         []AttnEntry    `json:"attn"`
	ConceptCount uint16         `json:"concept_count"`
	Concepts     []ConceptEntry `json:"concepts"`
	CRC          uint32         `json:"crc32"`
	rawBody      []byte
}

// Parser reads frames from a stream in lockstep.
type Parser struct{}

var (
	ErrBadMagic   = errors.New("bad magic")
	ErrBadVersion = errors.New("unsupported version")
	ErrCRC        = errors.New("crc mismatch")
	ErrShort      = errors.New("short read")
)

// ReadFrame reads one FrameV1 from r.
func (p *Parser) ReadFrame(r io.Reader) (*FrameV1, error) {
	hdr := make([]byte, 8)
	if _, err := io.ReadFull(r, hdr); err != nil {
		return nil, err
	}
	magic := binary.LittleEndian.Uint32(hdr[0:4])
	if magic != MagicMONO {
		return nil, ErrBadMagic
	}
	version := binary.LittleEndian.Uint16(hdr[4:6])
	if version != VersionV1 {
		return nil, ErrBadVersion
	}
	headerLen := binary.LittleEndian.Uint16(hdr[6:8])
	if headerLen < 8 {
		return nil, ErrShort
	}

	rest := make([]byte, int(headerLen)-8)
	if _, err := io.ReadFull(r, rest); err != nil {
		return nil, err
	}
	crcb := make([]byte, 4)
	if _, err := io.ReadFull(r, crcb); err != nil {
		return nil, err
	}
	crcRead := binary.LittleEndian.Uint32(crcb)

	crc := crc32.ChecksumIEEE(append(hdr, rest...))
	if crc != crcRead {
		return nil, ErrCRC
	}

	f := &FrameV1{
		Magic:     magic,
		Version:   version,
		HeaderLen: headerLen,
		rawBody:   append(hdr, rest...),
		CRC:       crcRead,
	}

	buf := rest
	off := 0
	get := func(n int) []byte {
		if off+n > len(buf) {
			off = len(buf) + 1
			return nil
		}
		b := buf[off : off+n]
		off += n
		return b
	}
	readU16 := func() uint16 { return binary.LittleEndian.Uint16(get(2)) }
	readU32 := func() uint32 { return binary.LittleEndian.Uint32(get(4)) }
	readU64 := func() uint64 { return binary.LittleEndian.Uint64(get(8)) }
	readHalf := func() Half { return Half(binary.LittleEndian.Uint16(get(2))) }

	f.PromptNonce = readU64()
	f.TokenIndex = readU32()
	f.ChosenID = readU32()
	f.TSC = readU64()

	f.TopK = readU16()
	f.TopLogits = make([]TopLogit, 0, f.TopK)
	for i := 0; i < int(f.TopK); i++ {
		tok := readU32()
		p16 := readHalf()
		f.TopLogits = append(f.TopLogits, TopLogit{TokenID: tok, Prob: p16.ToFloat32()})
	}

	f.AttnCount = readU16()
	f.Attn = make([]AttnEntry, 0, f.AttnCount)
	for i := 0; i < int(f.AttnCount); i++ {
		l := readU16()
		h := readU16()
		ti := readU16()
		w := readHalf().ToFloat32()
		f.Attn = append(f.Attn, AttnEntry{Layer: l, Head: h, TopTokenIdx: ti, Weight: w})
	}

	f.ConceptCount = readU16()
	f.Concepts = make([]ConceptEntry, 0, f.ConceptCount)
	for i := 0; i < int(f.ConceptCount); i++ {
		cid := readU16()
		score := readHalf().ToFloat32()
		f.Concepts = append(f.Concepts, ConceptEntry{ConceptID: cid, Score: score})
	}

	if off > len(rest) {
		return nil, ErrShort
	}
	return f, nil
}

