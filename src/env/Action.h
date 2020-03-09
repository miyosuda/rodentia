// -*- C++ -*-
#ifndef ACTION_HEADER
#define ACTION_HEADER

class Action {
public:
    int look;   // look left=[+], look right=[-]
    int strafe; // strafe left=[+1], strafe right=[-1]
    int move;   // forward=[+1], backward=[-1]

    Action()
        :
        look(0),
        strafe(0),
        move(0) {
    }

    Action(int look_, int strafe_, int move_)
        :
        look(look_),
        strafe(strafe_),
        move(move_) {
    }

    void set(int look_, int strafe_, int move_) {
        look   = look_;
        strafe = strafe_;
        move   = move_;
    }

    static int getActionSize() {
        return 3;
    }
};


#endif
